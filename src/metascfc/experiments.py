import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, KFold

from . import losses as loss_module
from . import metrics as metric_module
from . import saliency as saliency_module
from .seed import set_seed


class PriorGuidedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.prior_type = config.get("prior_type", "none")

        self.lambda_node = config.get("lambda_node", 0.0)
        self.lambda_module = config.get("lambda_module", 0.0)
        self.lambda_edge = config.get("lambda_edge", 0.0)

        self.task = config.get("task", "classification")
        self.alpha_aux = config.get("alpha_aux", 0.1)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.n_epochs = config.get("n_epochs", 100)

        self.roi_prior: Optional[np.ndarray] = None
        self.module_prior: Optional[np.ndarray] = None
        self.edge_prior: Optional[np.ndarray] = None
        self.roi_to_module: Optional[np.ndarray] = None

        self.history = {"train_loss": [], "val_metric": []}
        self.target_mean = 0.0
        self.target_std = 1.0

    def set_target_scaler(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Fit on the training fold only. Used only for regression."""
        self.target_mean = float(mean)
        self.target_std = float(std) if float(std) > 1e-8 else 1.0

    def _scale_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.task != "regression":
            return target
        return (target - self.target_mean) / self.target_std

    def _unscale_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        if self.task != "regression":
            return pred
        return pred * self.target_std + self.target_mean

    def set_priors(
        self,
        roi_prior: Optional[np.ndarray] = None,
        module_prior: Optional[np.ndarray] = None,
        edge_prior: Optional[np.ndarray] = None,
        roi_to_module: Optional[np.ndarray] = None,
    ) -> None:
        self.roi_prior = roi_prior
        self.module_prior = module_prior
        self.edge_prior = edge_prior
        self.roi_to_module = roi_to_module

    def _task_loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return loss_module.task_loss_classification(pred, target.long())
        return loss_module.task_loss_regression(pred, target.float())

    def _coupling_node_saliency(self, output: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Return [B,N] node saliency from coupling_vector or interaction matrix O."""
        if output.get("coupling_vector") is not None:
            cv = output["coupling_vector"]
            if cv.dim() == 1:
                cv = cv.unsqueeze(0)
            return saliency_module.node_saliency_from_coupling_vector(cv, mode="minmax_abs")
        if output.get("O") is not None:
            fc_sal, sc_sal = saliency_module.aggregate_node_saliency_from_interactions(output["O"], mode="mean_abs")
            node_sal = 0.5 * (fc_sal + sc_sal)
            return saliency_module.node_saliency_from_coupling_vector(node_sal, mode="minmax_abs")
        return None

    def _compute_prior_losses(
        self,
        output: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        node_loss = None
        module_loss = None
        edge_loss = None

        node_saliency = self._coupling_node_saliency(output)

        if self.prior_type in ("node", "combined") and self.roi_prior is not None and node_saliency is not None:
            prior_t = torch.from_numpy(self.roi_prior).float().to(self.device)
            node_loss = loss_module.pearson_corr_loss(node_saliency.mean(dim=0), prior_t)

        if self.prior_type in ("module", "combined") and self.module_prior is not None:
            if self.roi_to_module is None:
                raise ValueError("module prior requested but roi_to_module mapping was not provided.")
            roi_to_module_t = torch.from_numpy(self.roi_to_module).long().to(self.device)
            prior_t = torch.from_numpy(self.module_prior).float().to(self.device)
            num_modules = int(prior_t.numel())

            if output.get("O") is not None:
                module_sal = saliency_module.aggregate_module_saliency(output["O"], roi_to_module_t, num_modules)
                # Convert [B,M,M] module coupling to [B,M] module involvement.
                module_vec = 0.5 * (module_sal.mean(dim=2) + module_sal.mean(dim=1))
            elif node_saliency is not None:
                module_vec = saliency_module.aggregate_module_saliency_from_vector(
                    node_saliency, roi_to_module_t, num_modules, agg=self.config.get("module_saliency_agg", "mean")
                )
            else:
                module_vec = None

            if module_vec is not None:
                module_vec = saliency_module.node_saliency_from_coupling_vector(module_vec, mode="minmax_abs")
                module_loss = loss_module.pearson_corr_loss(module_vec.mean(dim=0), prior_t)

        if self.prior_type in ("edge", "combined") and self.edge_prior is not None:
            edge_prior_t = torch.from_numpy(self.edge_prior).float().to(self.device)
            if output.get("O") is not None:
                edge_sal = output["O"].abs().mean(dim=0)  # [N,N]
                edge_loss = loss_module.pearson_corr_loss(edge_sal, edge_prior_t)
            elif node_saliency is not None:
                # MS-Inter-GCN learns only corresponding ROI FC_i-SC_i coupling.
                # Therefore the only scientifically valid edge prior available here is
                # the diagonal/corresponding-edge prior, not a full cross-ROI N x N prior.
                diag_prior = torch.diag(edge_prior_t)
                edge_loss = loss_module.pearson_corr_loss(node_saliency.mean(dim=0), diag_prior)

        return node_loss, module_loss, edge_loss

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            fc_x = batch["fc_x"].to(self.device)
            fc_edge_index = batch["fc_edge_index"].to(self.device)
            fc_edge_weight = batch.get("fc_edge_weight")
            if fc_edge_weight is not None:
                fc_edge_weight = fc_edge_weight.to(self.device)
            sc_x = batch["sc_x"].to(self.device)
            sc_edge_index = batch["sc_edge_index"].to(self.device)
            sc_edge_weight = batch.get("sc_edge_weight")
            if sc_edge_weight is not None:
                sc_edge_weight = sc_edge_weight.to(self.device)
            y = batch["y"].to(self.device)
            fc_batch = batch.get("fc_batch")
            if fc_batch is not None:
                fc_batch = fc_batch.to(self.device)
            sc_batch = batch.get("sc_batch")
            if sc_batch is not None:
                sc_batch = sc_batch.to(self.device)

            optimizer.zero_grad()

            output = self.model(
                fc_x=fc_x,
                fc_edge_index=fc_edge_index,
                fc_edge_weight=fc_edge_weight,
                sc_x=sc_x,
                sc_edge_index=sc_edge_index,
                sc_edge_weight=sc_edge_weight,
                batch_fc=fc_batch,
                batch_sc=sc_batch,
            )

            y_for_loss = self._scale_target(y.float()) if self.task == "regression" else y
            task_loss = self._task_loss_fn(output["y_pred"], y_for_loss)
            aux_loss = loss_module.auxiliary_task_loss(
                output.get("fc_pred"), output.get("sc_pred"),
                y_for_loss, self.alpha_aux, self.task,
            )
            node_loss, module_loss, edge_loss = self._compute_prior_losses(output)
            prior_loss = loss_module.total_prior_loss(
                node_loss, module_loss, edge_loss,
                self.lambda_node, self.lambda_module, self.lambda_edge,
            ).to(self.device)

            loss = task_loss + aux_loss + prior_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        all_preds = []
        all_targets = []
        all_coupling = []
        all_O = []
        all_probs = []

        for batch in dataloader:
            fc_x = batch["fc_x"].to(self.device)
            fc_edge_index = batch["fc_edge_index"].to(self.device)
            fc_edge_weight = batch.get("fc_edge_weight")
            if fc_edge_weight is not None:
                fc_edge_weight = fc_edge_weight.to(self.device)
            sc_x = batch["sc_x"].to(self.device)
            sc_edge_index = batch["sc_edge_index"].to(self.device)
            sc_edge_weight = batch.get("sc_edge_weight")
            if sc_edge_weight is not None:
                sc_edge_weight = sc_edge_weight.to(self.device)
            y = batch["y"].to(self.device)
            fc_batch = batch.get("fc_batch")
            if fc_batch is not None:
                fc_batch = fc_batch.to(self.device)
            sc_batch = batch.get("sc_batch")
            if sc_batch is not None:
                sc_batch = sc_batch.to(self.device)

            output = self.model(
                fc_x=fc_x,
                fc_edge_index=fc_edge_index,
                fc_edge_weight=fc_edge_weight,
                sc_x=sc_x,
                sc_edge_index=sc_edge_index,
                sc_edge_weight=sc_edge_weight,
                batch_fc=fc_batch,
                batch_sc=sc_batch,
            )

            if self.task == "classification":
                preds = output["y_pred"].argmax(dim=1)
                probs = torch.softmax(output["y_pred"], dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())
            else:
                preds = self._unscale_prediction(output["y_pred"].view(-1))

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

            if output.get("coupling_vector") is not None:
                all_coupling.append(output["coupling_vector"].cpu().numpy())
            if output.get("O") is not None:
                all_O.append(output["O"].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs) if all_probs else None
        all_coupling = np.concatenate(all_coupling) if all_coupling else None
        all_O = np.concatenate(all_O) if all_O else None

        metrics = metric_module.compute_prediction_metrics(
            all_targets, all_preds, self.task, all_probs,
        )

        result = {
            "metrics": metrics,
            "predictions": all_preds,
            "targets": all_targets,
            "coupling": all_coupling,
            "O": all_O,
        }
        if all_probs is not None:
            result["probabilities"] = all_probs
        return result

    def train_and_evaluate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_metric = -float("inf")
        best_state = None

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_result = self.evaluate(val_loader)

            val_metric = val_result["metrics"].get("pearson", val_result["metrics"].get("accuracy", 0.0))
            self.history["train_loss"].append(train_loss)
            self.history["val_metric"].append(val_metric)

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        val_result = self.evaluate(val_loader)

        aux: Dict[str, Any] = {"prior_alignment": {}, "history": self.history}

        node_saliency = None
        if val_result["coupling"] is not None:
            cv_mean = val_result["coupling"].mean(axis=0)
            node_saliency = saliency_module.coupling_vector_to_saliency_vector(
                torch.from_numpy(cv_mean), mode="minmax_abs"
            ).numpy()
        elif val_result["O"] is not None:
            O_mean = torch.from_numpy(val_result["O"].mean(axis=0))
            fc_sal, sc_sal = saliency_module.aggregate_node_saliency_from_interactions(O_mean, mode="mean_abs")
            node_saliency = saliency_module.coupling_vector_to_saliency_vector(0.5 * (fc_sal + sc_sal), mode="minmax_abs").numpy()
        aux["node_saliency"] = node_saliency

        if node_saliency is not None and self.roi_prior is not None:
            aux["prior_alignment"]["node"] = metric_module.compute_prior_alignment_metrics(
                node_saliency, self.roi_prior, topk=self.config.get("topk", 10),
            )

        if node_saliency is not None and self.module_prior is not None and self.roi_to_module is not None:
            module_sal = saliency_module.aggregate_module_saliency_from_vector(
                torch.from_numpy(node_saliency).float(),
                torch.from_numpy(self.roi_to_module).long(),
                len(self.module_prior),
                agg=self.config.get("module_saliency_agg", "mean"),
            ).numpy()
            aux["module_saliency"] = module_sal
            aux["prior_alignment"]["module"] = metric_module.compute_prior_alignment_metrics(
                module_sal, self.module_prior, topk=min(self.config.get("topk", 10), len(self.module_prior)),
            )

        if self.edge_prior is not None:
            if val_result["O"] is not None:
                edge_sal = np.abs(val_result["O"]).mean(axis=0)
                aux["edge_saliency"] = edge_sal
                aux["prior_alignment"]["edge"] = metric_module.compute_prior_alignment_metrics(
                    edge_sal.flatten(), self.edge_prior.flatten(), topk=min(self.config.get("topk", 10), self.edge_prior.size),
                )
            elif node_saliency is not None:
                diag_prior = np.diag(self.edge_prior)
                aux["edge_saliency"] = node_saliency
                aux["prior_alignment"]["edge_diagonal"] = metric_module.compute_prior_alignment_metrics(
                    node_saliency, diag_prior, topk=self.config.get("topk", 10),
                )

        return val_result["metrics"], aux


def run_experiment(
    config: Dict[str, Any],
    dataset: torch.utils.data.Dataset,
    model_fn: Callable,
    device: torch.device,
    output_dir: str | Path,
    n_folds: int = 5,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = len(dataset)
    indices = np.arange(n_subjects)
    labels = np.array([dataset[i]["y"] for i in range(n_subjects)])

    if config.get("task") == "classification" and len(np.unique(labels)) >= 2:
        fold_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get("seed", 42))
    else:
        fold_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=config.get("seed", 42))

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splitter.split(indices, labels)):
        set_seed(config.get("seed", 42) + fold)

        model = model_fn().to(device)
        trainer = PriorGuidedTrainer(model, config, device)

        if config.get("prior_type", "none") != "none":
            trainer.set_priors(
                roi_prior=np.array(config.get("roi_prior")) if config.get("roi_prior") is not None else None,
                module_prior=np.array(config.get("module_prior")) if config.get("module_prior") is not None else None,
                edge_prior=np.array(config.get("edge_prior")) if config.get("edge_prior") is not None else None,
                roi_to_module=np.array(config.get("roi_to_module")) if config.get("roi_to_module") is not None else None,
            )

        train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), batch_size=config.get("batch_size", 8))
        val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), batch_size=config.get("batch_size", 8))

        metrics, aux = trainer.train_and_evaluate(train_loader, val_loader)
        fold_results.append({"metrics": metrics, "aux": aux})

    aggregated = aggregate_fold_results(fold_results)
    aggregated["config"] = config

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    return aggregated


def _flatten_alignment(aux: Dict[str, Any]) -> Dict[str, float]:
    flat = {}
    for scope, metrics in aux.get("prior_alignment", {}).items():
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                flat[f"prior_alignment_{scope}_{name}"] = float(value)
    return flat


def aggregate_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_metrics = [fr["metrics"] for fr in fold_results]
    metric_keys = all_metrics[0].keys()

    aggregated: Dict[str, Any] = {"fold_metrics": all_metrics}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    flat_alignments = [_flatten_alignment(fr.get("aux", {})) for fr in fold_results]
    all_alignment_keys = sorted({k for d in flat_alignments for k in d.keys()})
    for key in all_alignment_keys:
        values = [d[key] for d in flat_alignments if key in d]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated

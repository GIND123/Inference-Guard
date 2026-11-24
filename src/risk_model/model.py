"""Multi-task ModernBERT risk classifier for inferential privacy detection.

Predicts binary inferability for age, location, occupation, and education.
Provides input embedding hooks for Integrated Gradients and temperature scaling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

ATTRIBUTES: Tuple[str, ...] = ("age", "location", "occupation", "education")
ATTR_TO_IDX: Dict[str, int] = {attr: idx for idx, attr in enumerate(ATTRIBUTES)}
IDX_TO_ATTR: Dict[int, str] = {idx: attr for idx, attr in enumerate(ATTRIBUTES)}


class ModernBertRiskClassifier(nn.Module):
    """4-head multi-task risk classifier based on ModernBERT-base."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_heads: int = 4,
        num_classes_per_head: int = 1,
        dropout_prob: float = 0.1,
        attn_implementation: str = "sdpa",
        backbone: Optional[nn.Module] = None,
        hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head

        if backbone is not None:
            self.bert = backbone
            resolved_hidden_size = hidden_size or getattr(
                backbone, "hidden_size", getattr(getattr(backbone, "config", None), "hidden_size", 768)
            )
            self.config = getattr(backbone, "config", None)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            try:
                self.bert = AutoModel.from_pretrained(
                    model_name,
                    config=self.config,
                    attn_implementation=attn_implementation,
                )
            except Exception:
                self.bert = AutoModel.from_pretrained(model_name, config=self.config)
            resolved_hidden_size = getattr(self.config, "hidden_size", 768)

        effective_dropout = (
            getattr(self.config, "classifier_dropout", None)
            or getattr(self.config, "hidden_dropout_prob", None)
            or dropout_prob
        )
        self.dropout = nn.Dropout(effective_dropout)

        self.classifiers = nn.ModuleList(
            [nn.Linear(resolved_hidden_size, num_classes_per_head) for _ in range(num_heads)]
        )

        self.temperatures = nn.ParameterList(
            [nn.Parameter(torch.ones(1), requires_grad=False) for _ in range(num_heads)]
        )

    def get_input_embeddings(self) -> Optional[nn.Embedding]:
        """Return the backbone input embedding layer for Integrated Gradients."""
        if hasattr(self.bert, "get_input_embeddings"):
            emb = self.bert.get_input_embeddings()
            if emb is not None:
                return emb
        if hasattr(self.bert, "embeddings"):
            if hasattr(self.bert.embeddings, "tok_embeddings"):
                return self.bert.embeddings.tok_embeddings
            return self.bert.embeddings
        return None

    def set_temperatures(self, temperatures: Sequence[float]) -> None:
        """Set calibrated temperatures for all heads."""
        if len(temperatures) != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} temperatures, got {len(temperatures)}")
        for i, t in enumerate(temperatures):
            if t <= 0.0:
                raise ValueError(f"Temperature must be strictly positive, got {t} for head {i}")
            self.temperatures[i].data = torch.tensor([float(t)], dtype=torch.float32)

    def get_temperatures(self) -> List[float]:
        """Return list of current temperature values."""
        return [round(float(p.item()), 6) for p in self.temperatures]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        apply_temperature: bool = False,
    ) -> List[torch.Tensor]:
        """Compute logits for all 4 heads.

        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_len].
            attention_mask: Mask tensor of shape [batch_size, seq_len].
            inputs_embeds: Embedding tensor of shape [batch_size, seq_len, hidden_size].
            apply_temperature: Whether to divide raw logits by head temperatures.

        Returns:
            List of 4 tensors, each of shape [batch_size, num_classes_per_head].
        """
        if (input_ids is None and inputs_embeds is None) or (
            input_ids is not None and inputs_embeds is not None
        ):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            )

        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            sequence_output = outputs
        else:
            raise TypeError(f"Unexpected output type from backbone: {type(outputs)}")

        cls_representation = sequence_output[:, 0, :]
        cls_representation = self.dropout(cls_representation)

        logits: List[torch.Tensor] = []
        for i, classifier in enumerate(self.classifiers):
            head_logits = classifier(cls_representation)
            if apply_temperature:
                head_logits = head_logits / self.temperatures[i]
            logits.append(head_logits)

        return logits

    def predict_proba(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        apply_temperature: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute sigmoid probability tensor [batch_size, 1] per attribute."""
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            apply_temperature=apply_temperature,
        )
        probs: Dict[str, torch.Tensor] = {}
        for i, attr in enumerate(ATTRIBUTES):
            head_logit = logits[i]
            if head_logit.shape[-1] == 1:
                probs[attr] = torch.sigmoid(head_logit)
            else:
                probs[attr] = torch.softmax(head_logit, dim=-1)[:, 1:2]
        return probs

    def save_pretrained(self, save_dir: Union[str, Path]) -> None:
        """Save model weights and architecture configuration."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "model.pt")
        config_data = {
            "model_name": self.model_name,
            "num_heads": self.num_heads,
            "num_classes_per_head": self.num_classes_per_head,
            "attributes": list(ATTRIBUTES),
            "temperatures": self.get_temperatures(),
        }
        with open(save_path / "risk_model_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_dir: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        backbone: Optional[nn.Module] = None,
    ) -> ModernBertRiskClassifier:
        """Instantiate model and load serialized weights and temperatures."""
        load_path = Path(load_dir)
        with open(load_path / "risk_model_config.json", "r", encoding="utf-8") as f:
            config_data = json.load(f)

        model = cls(
            model_name=config_data["model_name"],
            num_heads=config_data["num_heads"],
            num_classes_per_head=config_data["num_classes_per_head"],
            backbone=backbone,
        )
        state_dict = torch.load(load_path / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        if "temperatures" in config_data:
            model.set_temperatures(config_data["temperatures"])
        model.to(device)
        return model


class FocalCrossEntropyLoss(nn.Module):
    """Focal loss addressing class imbalance without destabilizing gradient norms."""

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 1.5, ignore_index: int = -100):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal cross entropy loss over valid non-ignored targets."""
        import torch.nn.functional as F
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultiTaskModernBERTRiskModel(nn.Module):
    """Shared ModernBERT-base backbone with attribute-specific attention pooling and multi-class heads."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        dropout_prob: float = 0.15,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        backbone: Optional[nn.Module] = None,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        from src.contracts.schemas import AttributeName
        from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY

        if backbone is not None:
            self.encoder = backbone
            resolved_h = hidden_size or getattr(backbone, "hidden_size", 768)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
            resolved_h = getattr(self.config, "hidden_size", 768)

        # Attribute-specific attention pooling so each head attends to its own relevant tokens
        self.attention_poolers = nn.ModuleDict({
            attr.value: nn.Sequential(
                nn.Linear(resolved_h, resolved_h // 2),
                nn.Tanh(),
                nn.Linear(resolved_h // 2, 1, bias=False),
            )
            for attr in AttributeName
        })

        self.classifiers = nn.ModuleDict({
            attr.value: nn.Sequential(
                nn.LayerNorm(resolved_h),
                nn.Dropout(dropout_prob),
                nn.Linear(resolved_h, resolved_h // 2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(resolved_h // 2, len(labels)),
            )
            for attr, labels in ATTRIBUTE_TAXONOMY.items()
        })

        self.loss_fns = nn.ModuleDict({
            attr.value: FocalCrossEntropyLoss(
                weight=class_weights.get(attr.value) if class_weights else None,
                gamma=1.5,
            )
            for attr in AttributeName
        })

    def forward_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task attribute logits directly from token embeddings."""
        from src.contracts.schemas import AttributeName

        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        seq_hidden = getattr(outputs, "last_hidden_state", outputs)  # [B, L, H]
        mask_expanded = attention_mask.unsqueeze(-1).bool()  # [B, L, 1]

        logits_dict: Dict[str, torch.Tensor] = {}
        for attr in AttributeName:
            key = attr.value
            attn_scores = self.attention_poolers[key](seq_hidden)  # [B, L, 1]
            attn_scores = attn_scores.masked_fill(~mask_expanded, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, L, 1]
            pooled = torch.sum(seq_hidden * attn_weights, dim=1)  # [B, H]
            logits_dict[key] = self.classifiers[key](pooled)
        return logits_dict

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Forward pass through input embeddings and multi-task heads."""
        from src.contracts.schemas import AttributeName

        try:
            param_device = next(self.parameters()).device
            if input_ids.device != param_device:
                input_ids = input_ids.to(param_device)
            if attention_mask.device != param_device:
                attention_mask = attention_mask.to(param_device)
        except StopIteration:
            pass

        embed_fn = getattr(self.encoder, "get_input_embeddings", None)
        if callable(embed_fn):
            embeds = embed_fn()(input_ids)
        elif hasattr(self.encoder, "embeddings"):
            embeds = self.encoder.embeddings(input_ids)
        else:
            raise AttributeError("Backbone has no recognizable input embedding layer.")

        logits_dict = self.forward_from_embeds(embeds, attention_mask)
        result: Dict[str, Any] = {"logits": logits_dict}

        if labels is not None:
            total_loss = torch.tensor(0.0, device=input_ids.device)
            per_attr_loss = {}
            for attr in AttributeName:
                key = attr.value
                if key in labels:
                    attr_loss = self.loss_fns[key](logits_dict[key], labels[key])
                    per_attr_loss[key] = attr_loss
                    total_loss = total_loss + attr_loss
            result["loss"] = total_loss
            result["per_attribute_loss"] = per_attr_loss
        return result


def _available_model_files(model_name_or_path: str) -> Set[str]:
    """Return the set of filenames available for a local dir or a Hub repo id.

    For a local directory this is the names of its immediate entries. For a
    non-local repo id this lazily queries ``huggingface_hub.HfApi().list_repo_files``
    (import kept INSIDE this function so the module imports with huggingface_hub
    absent). If the listing cannot be obtained the available-file set is treated
    as unknown (empty) and the caller falls through gracefully."""
    model_path = Path(model_name_or_path)
    if model_path.exists() and model_path.is_dir():
        try:
            return {p.name for p in model_path.iterdir()}
        except Exception:
            return set()
    # Non-local: try the Hub. Keep the import lazy/local (sandbox has no hub dep).
    try:
        from huggingface_hub import HfApi

        return set(HfApi().list_repo_files(model_name_or_path))
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Could not list repo files for '%s' (%s: %s); treating file set as unknown.",
            model_name_or_path, type(exc).__name__, exc,
        )
        return set()


def _resolve_repo_file(model_name_or_path: str, filename: str) -> Optional[str]:
    """Resolve ``filename`` to a local path, downloading from the Hub if needed.

    Returns the filesystem path as a string, or ``None`` if it cannot be
    obtained. The ``huggingface_hub`` import is lazy/local."""
    model_path = Path(model_name_or_path)
    if model_path.exists() and model_path.is_dir():
        candidate = model_path / filename
        return str(candidate) if candidate.exists() else None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=model_name_or_path, filename=filename)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Could not download '%s' from '%s' (%s: %s).",
            filename, model_name_or_path, type(exc).__name__, exc,
        )
        return None


def _load_multitask_modernbert(
    model_name_or_path: str,
    device: str = "cpu",
    base_model: str = "answerdotai/ModernBERT-base",
) -> "MultiTaskModernBERTRiskModel":
    """Load the hosted multitask ModernBERT risk model (multitask_heads.pt layout).

    Handles both a local directory and a Hugging Face Hub repo id. Head sizes come
    from ATTRIBUTE_TAXONOMY (via the MultiTaskModernBERTRiskModel constructor); an
    optional ``multitask_config.json`` sidecar may override the base model. Reads
    ``calibration_params.json`` when present and attaches the parsed temperatures to
    the returned model as ``model._calibration_params`` for predict.py to consume.

    Raises if the state dict does not match this architecture at all, so that
    format detection in load_risk_classifier falls through rather than returning a
    mis-loaded model."""
    # Optional sidecar config (FEAT-002 will start uploading it; absent today).
    resolved_base_model = base_model
    sidecar_path = _resolve_repo_file(model_name_or_path, "multitask_config.json")
    if sidecar_path is not None:
        sidecar: Optional[Dict[str, Any]] = None
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Ignoring unreadable multitask_config.json for '%s' (%s: %s).",
                model_name_or_path, type(exc).__name__, exc,
            )
            sidecar = None
        if isinstance(sidecar, dict):
            resolved_base_model = (
                sidecar.get("base_model")
                or sidecar.get("model_name")
                or resolved_base_model
            )
            # Validate declared head sizes against the live taxonomy BEFORE loading
            # the state dict, so a taxonomy/weights drift surfaces as a named,
            # actionable error instead of a shape error that the caller swallows
            # into the untrained fallback. Only validate when the sidecar actually
            # declares num_classes_per_head; the current no-sidecar repo is untouched.
            num_classes_per_head = sidecar.get("num_classes_per_head")
            if isinstance(num_classes_per_head, dict):
                from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY

                expected = {
                    attr.value: len(labels)
                    for attr, labels in ATTRIBUTE_TAXONOMY.items()
                }
                for attr_name, sidecar_size in num_classes_per_head.items():
                    if attr_name in expected and sidecar_size != expected[attr_name]:
                        raise ValueError(
                            f"multitask_config.json head size mismatch for "
                            f"'{attr_name}': sidecar={sidecar_size} but current "
                            f"ATTRIBUTE_TAXONOMY={expected[attr_name]}; the hosted "
                            "weights were trained under a different taxonomy - "
                            "re-align the taxonomy or re-upload"
                        )

    weights_path = _resolve_repo_file(model_name_or_path, "multitask_heads.pt")
    if weights_path is None:
        raise FileNotFoundError(
            f"multitask_heads.pt could not be resolved for '{model_name_or_path}'."
        )

    model = MultiTaskModernBERTRiskModel(model_name=resolved_base_model)

    loaded = torch.load(weights_path, map_location=device)
    # Defensively unwrap a top-level 'state_dict' and strip 'module.' prefixes.
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        loaded = loaded["state_dict"]
    state_dict = {}
    for key, value in loaded.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        state_dict[new_key] = value

    result = model.load_state_dict(state_dict, strict=False)
    # If none of the architecture-defining keys loaded, the weights do not match
    # this model; raise so detection falls through to the next branch.
    missing = set(getattr(result, "missing_keys", []) or [])
    model_keys = set(model.state_dict().keys())
    loaded_arch_keys = [
        k for k in model_keys
        if ("classifiers" in k or "attention_poolers" in k) and k not in missing
    ]
    if not loaded_arch_keys:
        raise ValueError(
            f"multitask_heads.pt from '{model_name_or_path}' did not match "
            "MultiTaskModernBERTRiskModel (no classifier/attention_pooler keys loaded)."
        )

    # Optional calibration params -> attach for predict.py to build the scaler.
    calib_path = _resolve_repo_file(model_name_or_path, "calibration_params.json")
    if calib_path is not None:
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                model._calibration_params = json.load(f)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Ignoring unreadable calibration_params.json for '%s' (%s: %s).",
                model_name_or_path, type(exc).__name__, exc,
            )

    model.to(device)
    model.eval()
    return model


def load_risk_classifier(
    model_name_or_path: str,
    device: str = "cpu",
    trust_remote_code: bool = True,
) -> Any:
    """Load a trained risk classifier from a local directory or Hugging Face Hub.

    Performs explicit, ordered format detection by inspecting available files:
      (a) local ModernBertRiskClassifier custom format (risk_model_config.json + model.pt),
      (b) multitask ModernBERT layout (multitask_heads.pt, local file OR in the Hub list),
      (c) RoBERTa / HF-standard (config.json + safetensors/bin),
      (d) a LOUD untrained fallback whose message names the files actually found.
    Returns the initialized model transferred to the specified target device."""
    model_path = Path(model_name_or_path)
    available = _available_model_files(model_name_or_path)
    is_local_dir = model_path.exists() and model_path.is_dir()

    # (a) Local ModernBertRiskClassifier custom format: risk_model_config.json (+ model.pt).
    if is_local_dir and (model_path / "risk_model_config.json").exists():
        model = ModernBertRiskClassifier(model_name=str(model_path))
        if hasattr(model, "to"):
            model = model.to(device)
        return model

    # (b) Multitask ModernBERT layout: multitask_heads.pt present (local or Hub).
    if "multitask_heads.pt" in available or (
        is_local_dir and (model_path / "multitask_heads.pt").exists()
    ):
        try:
            return _load_multitask_modernbert(model_name_or_path, device=device)
        except ValueError as exc:
            # ValueError is the architecture-mismatch signal from
            # _load_multitask_modernbert (no classifier/attention_pooler keys
            # loaded, or a sidecar head-size mismatch): these weights genuinely
            # are not this format, so fall through to try RoBERTa and then the
            # loud untrained fallback. Log so this is debuggable.
            logging.getLogger(__name__).warning(
                "multitask_heads.pt detected for '%s' but could not be loaded (%s: %s); "
                "falling through to other formats.",
                model_name_or_path, type(exc).__name__, exc,
            )
        except Exception as exc:
            # Any other exception (ImportError, ModuleNotFoundError, RuntimeError,
            # OSError, FileNotFoundError, ...) means the weights ARE the detected
            # multitask format but the ENVIRONMENT/LOADING failed (e.g. a broken
            # torchvision breaks transformers' lazy vision imports, or a download
            # failed). Silently degrading to an UNTRAINED model would serve random
            # risk scores, so fail fast with the real cause instead of falling
            # through. (ValueError == "not this format"; everything else == a
            # detected-but-unloadable trained checkpoint.)
            logging.getLogger(__name__).error(
                "multitask_heads.pt detected for '%s' but could not be loaded due to an "
                "environment/dependency error (%s: %s); refusing to fall back to an "
                "untrained model.",
                model_name_or_path, type(exc).__name__, exc,
            )
            raise RuntimeError(
                f"Detected a trained multitask risk checkpoint (multitask_heads.pt) for "
                f"'{model_name_or_path}' but could not load it due to an "
                f"environment/dependency error: {exc}. Refusing to fall back to an untrained "
                f"model. Fix the environment (e.g. a broken torchvision breaks transformers' "
                f"lazy vision imports) and retry."
            ) from exc

    # (c) RoBERTa / HF-standard format.
    try:
        from src.risk_model.roberta_model import RobertaRiskClassifier, RobertaRiskConfig
        model = RobertaRiskClassifier.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if hasattr(model, "to"):
            model = model.to(device)
        return model
    except Exception as exc:
        # (d) This fallback builds an untrained model. It used to be silent, and
        # the product served random heads without anyone noticing (Session 10).
        found = sorted(available) if available else "unknown (could not list files)"
        logging.getLogger(__name__).error(
            "Could not load risk classifier '%s' (%s: %s); found files: %s. Falling back to "
            "an UNTRAINED RobertaRiskClassifier: risk scores from it are not meaningful.",
            model_name_or_path, type(exc).__name__, exc, found,
        )
        from src.risk_model.roberta_model import RobertaRiskClassifier, RobertaRiskConfig
        config = RobertaRiskConfig(base_model_name=model_name_or_path)
        model = RobertaRiskClassifier(config=config)
        if hasattr(model, "to"):
            model = model.to(device)
        return model


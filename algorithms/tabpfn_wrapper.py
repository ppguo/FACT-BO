import torch
import numpy as np
from tabpfn.model.loading import load_model
from tabpfn.model.config import InferenceConfig
from tabpfn.base import initialize_tabpfn_model

class VanillaDirectTabPFNRegressor:
    """
    Minimal "direct" TabPFN regressor wrapper for inference-only usage.
    Uses the real `load_model` function to load a pretrained TabPFN model.
    """

    def __init__(
        self,
        model_path: str = "auto",
        fit_mode: str = "fit_preprocessors",
        device: str = "auto",
        inference_precision: str = "auto",
        random_state=None,
    ):
        self.model_path = model_path
        self.fit_mode = fit_mode
        self.device = device
        self.inference_precision = inference_precision
        self.random_state = random_state

        # 1) Static seed for reproducibility
        static_seed, _ = self._infer_random_state(self.random_state)

        # 2) Load the pretrained TabPFN model, config, and bar distribution
        self.model_, self.config_, self.bardist_ = self._initialize_tabpfn_model(
            model_path=self.model_path,
            which="regression",    # TabPFN's recognized keyword is "regression"
            fit_mode=self.fit_mode,
            static_seed=static_seed,
        )

        # 3) Choose device + set precision 
        self.device_ = self._infer_device_and_type(self.device)

        self.model_.to(self.device_)
        self.bardist_.borders = self.bardist_.borders.to(self.device_)


        self.use_autocast_, self.forced_inference_dtype_, byte_size = self._determine_precision(
            self.inference_precision,
            self.device_,
        )

        # Move the loaded model to device, set to eval mode
        self.model_.to(self.device_)
        self.model_.eval()


    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        single_eval_pos: int,
    ) -> dict:
        """
        Forward pass for inference with shape [time, batch, features] for X
        and [time, batch] for Y.
        """
        # Move to device
        X = X.to(self.device_)
        Y = Y.to(self.device_)

        # Suppose Y has shape [time, batch, 1] 
        # We want to compute mean & std only for the "train" portion 
        # i.e. time steps [0 .. single_eval_pos-1]
        y_train_part = Y[:single_eval_pos]  # shape: [single_eval_pos, batch, 1]

        y_mean = y_train_part.mean(dim=0, keepdim=True)
        y_std = y_train_part.std(dim=0, keepdim=True) + 1e-9  # add epsilon
        self.y_mean = y_mean
        self.y_std = y_std

        # Now standardize the train portion
        Y_eval = Y.clone().to(self.device_)
        Y_eval[:single_eval_pos,:,:] = (Y[:single_eval_pos,:,:] - y_mean) / y_std

        # Operate under the mode where we can get gradients
        output_dict = self.model_(
            None,        # style is None
            X,           # shape: (time, batch, features)
            Y_eval,           # shape: (time, batch)
            single_eval_pos=single_eval_pos,
            only_return_standard_out=False,
        )
        return output_dict 

    def compute_loss(self, logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        If your loaded model is a regression distribution model using bar distributions,
        you can compute the negative log-likelihood (NLL) via self.bardist_.
        """
        Y = Y.to(self.device_)
        logits = logits.to(self.device_)
        self.bardist_ = self.bardist_.to(self.device_)
        loss = self.bardist_(logits, Y)
        return loss.mean()

    def predict_mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        For the distribution-based model, return the predicted mean from the bar distribution.
        """
        mean_pred = self.bardist_.mean(logits)
        mean_pred = mean_pred * self.y_std.squeeze(-1) + self.y_mean.squeeze(-1)
        return mean_pred
    
    def predict_variance(self, logits: torch.Tensor) -> torch.Tensor:
        """
        For the distribution-based model, return the predicted variance from the bar
        distribution, in the SAME physical units as ``predict_mean``.

        FIX I-004: ``forward()`` standardizes the train portion of Y before the model call, so
        the bar distribution's borders -- and therefore ``bardist_.variance`` -- live in
        standardized space. ``predict_mean`` already undoes that with ``* y_std + y_mean``;
        variance needs ``* y_std**2``. Production returned the standardized variance, so
        SamplingUCB sampled from ``Normal(mu_raw, sigma_standardized)`` and its exploration
        strength scaled wrongly with the target's physical magnitude.
        """
        var_pred = self.bardist_.variance(logits)
        return var_pred * self.y_std.squeeze(-1).square()
    
    def predict_ei(self, logits: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
        y_mean = self.y_mean.squeeze(-1)
        y_std = self.y_std.squeeze(-1)
        standardized_best = ((best_f - y_mean) / y_std).expand(logits.shape[:-1])
        standardized_ei = self.bardist_.ei(logits, standardized_best)
        return standardized_ei * y_std

    def predict_ucb(self, logits: torch.Tensor, best_f: torch.Tensor, percentile=0.05) -> torch.Tensor:
        standardized_ucb = self.bardist_.ucb(
            logits=logits, best_f=best_f, rest_prob=percentile
        )
        return standardized_ucb * self.y_std.squeeze(-1) + self.y_mean.squeeze(-1)

    def _infer_random_state(self, random_state):
        """
        Example helper that returns (static_seed, rng).
        """
        if random_state is None:
            random_state = 0
        rng = np.random.default_rng(random_state)
        return random_state, rng

    def _infer_device_and_type(self, device):
        """
        Picks device. E.g. "auto" => cuda if available, else cpu.
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _determine_precision(self, inference_precision, device):
        """
        Returns (use_autocast, forced_dtype, byte_size).
        Adjust if you have your own logic for half-precision vs float32, etc.
        """
        if inference_precision == "auto":
            # default: use mixed-precision if CUDA, else float32
            if device.type == "cuda":
                return (True, None, 2)
            else:
                return (False, None, 4)
        elif inference_precision == "float32":
            return (False, torch.float32, 4)
        elif inference_precision == "float16":
            return (False, torch.float16, 2)
        return (False, None, 4)

    def _initialize_tabpfn_model(self, model_path, which, fit_mode, static_seed):
        # This is a simplified version of the initialize_tabpfn_model function
        from tabpfn.base import initialize_tabpfn_model
        return initialize_tabpfn_model(
            model_path=model_path,
            which=which,
            fit_mode=fit_mode,
            static_seed=static_seed,
        )



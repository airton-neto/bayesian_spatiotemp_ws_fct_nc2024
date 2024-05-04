# %%
import numpy as np
import properscoring as ps
import torch
from torch.functional import F

# from swag.posteriors import SWAG
# from understandingbdl.swag.posteriors.swag import SWAG
from subspace_inference.posteriors.swag import SWAG
from swag import utils
from swag.models.custom import (
    ConvLSTMFullModel,
    ConvLSTMNWPOnly,
    DummyFullModel,
    GaussianBase,
    LSTMFullModel,
    LSTMWTGOnly,
    MLPFullModel,
)


# %%
def factory(model_base):
    def build_model(
        dataset, gaussian=False, swag=False, dropout_rate=0.1, n_hidden=64
    ):
        mymodelargs = dict(
            num_classes=dataset[0][1].shape[1],
            fct_input_size=dataset[0][0][1].shape[1],
            wtg_input_size=dataset[0][0][0].shape[1],
            hidden_size=n_hidden,  # Testar 64, 128
            num_layers=2,
            seq_length=dataset[0][0][0].shape[0],
            dropout_r=dropout_rate,
        )
        if gaussian:
            model = GaussianBase(model_base, **mymodelargs)
        elif swag:
            # model = SWAG(
            #     model_base, no_cov_mat=True, max_num_models=100, **mymodelargs
            # )
            model = SWAG(
                model_base,
                subspace_type="pca",
                **mymodelargs,
                subspace_kwargs={"max_rank": 10, "pca_rank": 10},
            )
        else:
            model = model_base(**mymodelargs)

        return model

    return build_model


build_convlstm_model = factory(ConvLSTMFullModel)
build_lstm_model = factory(LSTMFullModel)
build_mlp_model = factory(MLPFullModel)
build_dummy_model = factory(DummyFullModel)
build_wtg_model = factory(LSTMWTGOnly)
build_nwp_model = factory(ConvLSTMNWPOnly)


# %%
def get_metrics(Y, means, stds):
    # 1. RMSE
    rmse = np.sqrt(F.mse_loss(means, Y).item())

    # 2. Negative Log-Likelihood (Gaussian)
    nll = F.gaussian_nll_loss(means, Y, stds).item()

    # 3. Continuous Ranked Probability Score
    crps = ps.crps_gaussian(
        Y.cpu().numpy(), mu=means.cpu().numpy(), sig=stds.cpu().numpy()
    ).mean()

    print("RMSE", rmse, "NLL", nll, "CRPS", crps)

    return rmse, nll, crps


# %%
class BayesianModelPredictor:
    def __init__(self, models, model_type, loader_scaler):
        self.models = models
        self.model_type = model_type
        self.loader_scaler = loader_scaler

    def _swag_sampler_method(self, model, loader, n_samples=25):
        model.sample(scale=10)
        init = utils.predict(
            loader=loader, model=model, loader_scaler=self.loader_scaler
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]

        # Outros samples
        for i in range(n_samples - 1):
            model.sample(scale=10)
            predictions = torch.concatenate(
                (
                    predictions,
                    utils.predict(
                        loader=loader,
                        model=model,
                        loader_scaler=self.loader_scaler,
                    )["predictions"][..., None],
                ),
                dim=3,
            )

        stds = torch.std(predictions, dim=3)
        means = torch.mean(predictions, dim=3)

        return Y, means, stds

    def _laplace(self, loader, n_samples=25):
        la = self.models[1]

        la.sample(scale=1.0, cov=False)
        init = utils.predict(
            loader=loader, model=la.net, loader_scaler=self.loader_scaler
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]

        # Outros samples
        for i in range(n_samples - 1):
            la.sample(scale=1.0, cov=False)
            predictions = np.concatenate(
                (
                    predictions,
                    utils.predict(loader=loader, model=la.net)["predictions"][
                        ..., None
                    ],
                ),
                axis=3,
            )

        stds = np.std(predictions, axis=3)
        means = np.mean(predictions, axis=3)

        return Y, means, stds

    def _swag(self, loader, n_samples=25):
        return self._swag_sampler_method(
            self.models[1],
            loader,
            n_samples=n_samples,
        )

    def _multiswag(self, loader, n_samples=20):
        Ys = []
        Ms = []
        STs = []
        for model in self.models[1:]:
            Y, means, stds = self._swag_sampler_method(
                model, loader, n_samples=n_samples
            )
            Ys.append(Y[..., None])
            Ms.append(means[..., None])
            STs.append(stds[..., None])
        Y = torch.concatenate(Ys, dim=3).mean(dim=3)
        means = torch.concatenate(Ms, dim=3).mean(dim=3)
        stds = torch.concatenate(STs, dim=3).mean(dim=3)

        return Y, means, stds

    def _dropout(self, loader, n_samples=25):
        init = utils.predict(
            loader=loader,
            model=self.models[0],
            use_training_true=True,
            loader_scaler=self.loader_scaler,
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]
        # Outros samples
        for i in range(n_samples - 1):
            predictions = torch.concatenate(
                (
                    predictions,
                    utils.predict(
                        loader=loader,
                        model=self.models[0],
                        use_training_true=True,
                        loader_scaler=self.loader_scaler,
                    )["predictions"][..., None],
                ),
                dim=3,
            )

        stds = torch.std(predictions, dim=3)
        means = torch.mean(predictions, dim=3)

        return Y, means, stds

    def _ensemble(self, loader):
        means_ = []
        variances_ = []
        for model in self.models:
            preds = utils.predict_gaussian(
                loader=loader, model=model, loader_scaler=self.loader_scaler
            )
            means, variances = preds["predictions"]
            means_.append(means[..., None])
            variances_.append(variances[..., None])

        means_f = torch.concatenate(means_, dim=3).mean(dim=3)
        variances_f = (
            (torch.concatenate(means_, dim=3) ** 2).mean(dim=3)
            + torch.concatenate(variances_, dim=3).mean(dim=3)
            - means_f**2
        )
        Y = preds["targets"]

        return Y, means_f, torch.sqrt(variances_f)

    def _dummy(self, loader, n_samples=3):
        init = utils.predict(
            loader=loader,
            model=self.models[0],
            loader_scaler=self.loader_scaler,
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]

        # Outros samples
        for i in range(n_samples - 1):
            predictions = torch.concatenate(
                (
                    predictions,
                    utils.predict(
                        loader=loader,
                        model=self.models[0],
                        loader_scaler=self.loader_scaler,
                    )["predictions"][..., None],
                ),
                dim=3,
            )

        stds = torch.std(predictions, dim=3)
        means = torch.mean(predictions, dim=3)

        return Y, means, stds

    def _nllbaseline(self, loader):
        model = self.models[0]
        preds = utils.predict_gaussian(
            loader=loader, model=model, loader_scaler=self.loader_scaler
        )
        means, variances = preds["predictions"]
        Y = preds["targets"]
        return Y, means, torch.sqrt(variances)

    def sample(self, loader, n_samples=25):
        if self.model_type == "swag":
            return self._swag(loader, n_samples=n_samples)
        elif self.model_type == "laplace":
            return self._laplace(loader, n_samples=n_samples)
        elif self.model_type == "multiswag":
            return self._multiswag(loader, n_samples=n_samples)
        elif self.model_type == "dropout":
            return self._dropout(loader, n_samples=n_samples)
        elif self.model_type == "ensemble":
            return self._ensemble(loader)
        elif self.model_type == "dummy":
            return self._dummy(loader)
        elif self.model_type == "nllbaseline":
            return self._nllbaseline(loader)
        else:
            raise Exception(
                f"O model_type {self.model_type} não tem Sampler associado"
            )

    def _ensemble_input(self, input):
        means_ = []
        variances_ = []
        with torch.no_grad():
            for model in self.models:
                means, variances = model(input)
                means_.append(means[..., None])
                variances_.append(variances[..., None])

        means_f = torch.concatenate(means_, dim=3).mean(dim=3)
        variances_f = (
            torch.concatenate(means_, dim=3).mean(dim=3) ** 2
            + torch.concatenate(variances_, dim=3).mean(dim=3)
            - means_f**2
        )

        return means_f, np.sqrt(variances_f)

    def _ensemble_input_sample(self, input):
        means, stds = self._ensemble_input(input[None, ...])
        return (
            means.cpu() + (torch.randn(list(means.shape)) * stds.cpu()).numpy()
        )[0, :, :]

    def simple_sample(self, Y, means, stds, loader=None):
        if loader is not None:
            with torch.no_grad():
                Y, means, stds = self.sample(loader, n_samples=25)
        else:
            pass
        sample = (
            means.cpu() + (torch.randn(list(means.shape)) * stds.cpu()).numpy()
        )
        return Y.cpu(), means.cpu(), sample

    def simple_sample_with_dataset(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            Y, means, stds = self.sample(loader, n_samples=25)
            sample = (
                means.cpu()
                + (torch.randn(list(means.shape)) * stds.cpu()).numpy()
            )
        return Y.cpu(), means.cpu(), sample.cpu()


# %%

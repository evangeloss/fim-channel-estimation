import io
import unittest
import numpy as np
import torch
from model.estimator import ChannelEstimator
from dataset.generator import generate_dataset


class RelativeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)
        self.model = ChannelEstimator(n_deformations=3,n_h_b=2,n_v_b=2,n_h_u=2,n_v_u=2,
            feature_channels=8,geometry_dim=8,geometry_hidden_dim=8,
            modulation_hidden_dim=8,decoder_hidden_channels=8,dilations=(1,2))
        self.y = torch.randn(2,12,4,4)
        self.g = self.model.nominal_geometry[None,None].repeat(2,3,1,1,1)
        self.g += torch.randn_like(self.g)*0.1

    def test_relative_inputs_preserve_observations_and_geometry(self):
        x,g = self.model.preprocessing(self.y,self.g,self.model.nominal_geometry)
        torch.testing.assert_close(x[:,:,0:4]+x[:,:,4:8],self.y.reshape(2,3,4,4,4))
        torch.testing.assert_close(g[...,:6]+g[...,6:12]+g[...,12:],self.g)
        self.assertEqual(torch.count_nonzero(x[:,0,4:]).item(),0)
        self.assertEqual(torch.count_nonzero(g[:,0,12:]).item(),0)

    def test_common_displacement_is_retained(self):
        common = self.model.nominal_geometry[None,None].repeat(2,3,1,1,1)+0.2
        _,g = self.model.preprocessing(self.y,common,self.model.nominal_geometry)
        torch.testing.assert_close(g[...,6:12],torch.full_like(g[...,6:12],0.2))
        self.assertEqual(torch.count_nonzero(g[...,12:]).item(),0)

    def test_zero_decoder_recovers_mean_exactly(self):
        torch.nn.init.zeros_(self.model.decoder.output_head.weight)
        torch.nn.init.zeros_(self.model.decoder.output_head.bias)
        pred = self.model(self.y,self.g)
        torch.testing.assert_close(pred,self.y.reshape(2,3,4,4,4).mean(1))

    def test_spatial_modulation_and_gradient_flow(self):
        pred,aux = self.model(self.y,self.g,True)
        self.assertEqual(aux['gamma'].shape,(2,3,8,4,4))
        self.assertGreater(aux['gamma'].var(dim=-1).sum().item(),0)
        pred.square().mean().backward()
        for branch in (self.model.cnn,self.model.geometry_encoder,self.model.modulation,self.model.decoder):
            grads = [p.grad for p in branch.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0)

    def test_disable_modulation_removes_geometry_dependence(self):
        a = self.model(self.y,self.g,modulation_strength=0)
        b = self.model(self.y,self.g+0.5,modulation_strength=0)
        torch.testing.assert_close(a,b)

    def test_joint_nonreference_permutation(self):
        order = [0,2,1]
        yp = self.y.reshape(2,3,4,4,4)[:,order].reshape_as(self.y)
        torch.testing.assert_close(self.model(self.y,self.g),self.model(yp,self.g[:,order]))

    def test_checkpoint_roundtrip(self):
        data = io.BytesIO()
        torch.save(dict(model_config=self.model.config,model_state_dict=self.model.state_dict()),data)
        data.seek(0)
        checkpoint = torch.load(data,weights_only=True)
        restored = ChannelEstimator(**checkpoint['model_config'])
        restored.load_state_dict(checkpoint['model_state_dict'])
        torch.testing.assert_close(self.model(self.y,self.g),restored(self.y,self.g))

    def test_raw_variant_forward(self):
        config = dict(self.model.config,variant='raw')
        self.assertEqual(ChannelEstimator(**config)(self.y,self.g).shape,(2,4,4,4))

    def test_dataset_target_unchanged(self):
        wl = 3e8/28e9
        kwargs = dict(n_h_b=2,n_v_b=2,d_x_b=wl/8,d_y_b=wl/8,
            n_h_u=2,n_v_u=2,d_x_u=wl/8,d_y_u=wl/8,fc=28e9,fs=1e5,
            n_subcarriers=2,n_deformations=3,n_channels=2,n_paths=3,
            snr_values=[20],b_min=.1,b_max=.1,seed=7)
        ym,g,tm = generate_dataset(**kwargs,residual_reference='mean')
        yf,gf,tf = generate_dataset(**kwargs,residual_reference='first')
        np.testing.assert_array_equal(ym,yf)
        np.testing.assert_array_equal(g,gf)
        np.testing.assert_allclose(ym.reshape(2,3,4,4,4).mean(1)+tm,yf[:,:4]+tf,atol=2e-7)
        loss = (self.model(torch.from_numpy(ym),torch.from_numpy(g))-
                torch.from_numpy(ym.reshape(2,3,4,4,4).mean(1)+tm)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()

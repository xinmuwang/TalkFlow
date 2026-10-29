"""Offline shape/interface smoke tests; do not require BEAT2 or pretrained weights."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.dataset import build_dataloader, PoseExprLatentDataset
from utils.speech_dataset import (
    build_speech_dataloader, build_long_test_dataloader,
)
from inference.flow import (
    make_sliding_240_windows, postprocess_generated_pose6d,
    save_all_gen_out_to_npz_safe,
)
from training.ae import ae_losses, evaluate_ae


def make_legacy_entry(T=300):
    return {
        'seq_id':'2_scott_test', 'T':T,
        'poses':np.zeros((T,165),dtype=np.float32),
        'expressions':np.zeros((T,100),dtype=np.float32),
        'hubert':np.zeros((T,1024),dtype=np.float32),
        'librosa':np.zeros((T,35),dtype=np.float32),
        'text_global':np.zeros(384,dtype=np.float32),
        'text_words_vecs':np.ones((2,300),dtype=np.float32),
        'text_words_starts':np.array([0.,2.],dtype=np.float32),
        'text_words_ends':np.array([0.5,3.],dtype=np.float32),
        'betas':np.zeros(300,dtype=np.float32),
        'trans':np.zeros((T,3),dtype=np.float32),
    }


def make_processed_dir(tmp_path: Path):
    root=tmp_path/'processed'; (root/'clips'/'train').mkdir(parents=True)
    (root/'sequences'/'test').mkdir(parents=True)
    (root/'manifests').mkdir(parents=True)
    clip='2_scott_test_0000.npz'; seq='2_scott_test.npz'
    np.savez_compressed(root/'clips'/'train'/clip, rot6d=np.zeros((240,55,6),dtype=np.float32),
         expression=np.zeros((240,100),dtype=np.float32),
         hubert=np.zeros((240,1024),dtype=np.float16),
         librosa35=np.zeros((240,35),dtype=np.float16),
         fasttext=np.ones((240,300),dtype=np.float16),
         text_local_mask=np.ones(240,dtype=np.bool_),
         sbert=np.zeros(384,dtype=np.float16), seq_id='2_scott_test',
         betas=np.zeros(300,dtype=np.float32))
    np.savez_compressed(root/'sequences'/'test'/seq, **{
       k:v for k,v in make_legacy_entry(300).items() if k not in ['hubert','librosa']},
       hubert_seq=np.zeros((180,1024),dtype=np.float16),
       hubert_fps=np.float32(18.),
       librosa_seq=np.zeros((300,35),dtype=np.float16),
       librosa_fps=np.float32(30.))
    (root/'manifests'/'train_processed.jsonl').write_text(json.dumps({
       'processed_path':'clips/train/'+clip,'seq_id':'2_scott_test'})+'\n')
    (root/'manifests'/'test_sequence_processed.jsonl').write_text(json.dumps({
       'processed_path':'sequences/test/'+seq,'seq_id':'2_scott_test'})+'\n')
    return root


def test_legacy_ae_dataset(tmp_path):
    f=tmp_path/'legacy.pkl'
    with f.open('wb') as fp:pickle.dump([make_legacy_entry()],fp)
    ds,loader=build_dataloader(f,batch_size=1,shuffle=False,clip_len=240,
                               return_betas=True,return_trans=True)
    b=next(iter(loader))
    assert b['pose'].shape==(1,240,55,6)
    assert b['expression'].shape==(1,240,100)
    assert b['betas'].shape==(1,300)
    assert b['trans'].shape==(1,240,3)


def test_npz_ae_and_speech_dataset(tmp_path):
    root=make_processed_dir(tmp_path)
    ds,loader=build_dataloader(root,batch_size=1,shuffle=False,split='train')
    b=next(iter(loader))
    assert b['pose'].shape==(1,240,55,6)
    _,flowloader=build_speech_dataloader(root,batch_size=1,shuffle=False,split='train')
    out=next(iter(flowloader))
    assert out['hubert'].shape==(1,240,1024)
    assert out['librosa'].shape==(1,240,35)
    assert out['text_local_mask'].sum()==240
    assert out['text_local_raw'].shape==(1,240,300)
    assert out['valid_mask'].shape==(1,240)


def test_sequence_test_and_saving(tmp_path):
    root=make_processed_dir(tmp_path)
    ds,loader=build_long_test_dataloader(root,split='test')
    b=next(iter(loader))
    assert b['hubert'].shape==(1,300,1024)
    assert b['librosa'].shape==(1,300,35)
    assert b['pose'].shape==(1,300,55,6)
    assert b['betas'].shape==(1,300)
    wins=make_sliding_240_windows(300)
    assert wins==[(0,240),(60,300)]
    pose=torch.zeros((300,55,6),dtype=torch.float32)
    post=postprocess_generated_pose6d(pose,wins,smooth_kernel_size=3,blend_radius=4)
    gen={'pose_sequences':[post],'expr_sequences':[torch.zeros((300,100))],
         'windows':[wins]}
    saved=save_all_gen_out_to_npz_safe(gen,loader,save_dir=tmp_path/'out')
    with np.load(saved[0],allow_pickle=False) as d:
        assert d['poses'].shape==(300,165)
        assert d['trans'].shape==(300,3)
        assert d['betas'].shape==(300,)


def test_long_audio_only_saving(tmp_path):
    root=make_processed_dir(tmp_path)
    _,loader=build_long_test_dataloader(root,load_gt=False,split='test')
    b=next(iter(loader))
    assert 'poses' not in b and 'pose' not in b
    gen={'pose_sequences':[torch.zeros((300,55,6))],
         'expr_sequences':[torch.zeros((300,100))]}
    saved=save_all_gen_out_to_npz_safe(gen,loader,save_dir=tmp_path/'out')
    with np.load(saved[0],allow_pickle=False) as d:
        assert d['trans'].shape==(300,3) and np.allclose(d['trans'],0)


def test_ae_loss_and_evaluation(tmp_path):
    f=tmp_path/'legacy.pkl'
    with f.open('wb') as fp:pickle.dump([make_legacy_entry(240)],fp)
    _,loader=build_dataloader(f,batch_size=1,shuffle=False,clip_len=240)
    class DummyAE(torch.nn.Module):
        def forward(self,p,e): return {'pose_recon':p,'expr_recon':e}
    out=evaluate_ae(DummyAE(),loader,torch.device('cpu'))
    assert all(abs(v)<1e-8 for v in out.values())


def test_current_flow_forward_shapes():
    # Real modules, random features; no AE/PyG or weights necessary.
    torch.set_num_threads(2)
    from configs.models import CONDITION_KWARGS,FLOW_KWARGS
    from modules.condition_encoder_3stream import ConditionEncoder3Stream
    from modules.flow_matching_backbone_3stream import FlowMatchingTransformerBackbone3Stream
    from inference.flow import sample_flow_matching_euler_3stream
    cond_model=ConditionEncoder3Stream(**CONDITION_KWARGS).eval()
    flow=FlowMatchingTransformerBackbone3Stream(**FLOW_KWARGS).eval()
    with torch.no_grad():
        c=cond_model(hubert=torch.zeros(1,240,1024),librosa=torch.zeros(1,240,35),
                text_local_raw=torch.zeros(1,240,300),text_local_mask=torch.zeros(1,240,dtype=torch.bool),
                text_global_raw=torch.zeros(1,384),valid_mask=torch.ones(1,240,dtype=torch.bool))
        assert c['cond_upper'].shape==(1,60,512)
        u,l,e=sample_flow_matching_euler_3stream(flow,c['cond_upper'],c['cond_lower'],c['cond_expr'],
                 c['cond_mask'],upper_latent_dim=512,lower_latent_dim=128,expr_latent_dim=512,num_steps=1)
        assert u.shape==(1,60,512) and l.shape==(1,60,128) and e.shape==(1,60,512)


def test_short_nonmultiple_inference_keeps_original_length(tmp_path):
    from inference.flow import generate_long_sequence_240_shift_last_flow_3stream
    f=tmp_path/'short.pkl'
    with f.open('wb') as o:
        pickle.dump([make_legacy_entry(229)],o)
    _,loader=build_long_test_dataloader(f)
    class FakeCond(torch.nn.Module):
        def forward(self,hubert,**kwargs):
            B,T,_=hubert.shape
            n=(T+3)//4
            return {'cond_upper':hubert.new_zeros(B,n,512),
                    'cond_lower':hubert.new_zeros(B,n,512),
                    'cond_expr':hubert.new_zeros(B,n,512),
                    'seq_mask':torch.ones(B,n,dtype=torch.bool)}
    class FakeFlow(torch.nn.Module):
        def forward(self,x_upper_t,x_lower_t,x_expr_t,**kwargs):
            return {'v_upper':torch.zeros_like(x_upper_t),
                    'v_lower':torch.zeros_like(x_lower_t),
                    'v_expr':torch.zeros_like(x_expr_t)}
    class FakeAE(torch.nn.Module):
        def decode_from_latents(self,z_upper,z_lower,z_expr):
            B,T,_=z_upper.shape
            return {'pose_recon':z_upper.new_zeros(B,T*4,55,6),
                    'expr_recon':z_upper.new_zeros(B,T*4,100)}
    result=generate_long_sequence_240_shift_last_flow_3stream(
        FakeAE(),FakeCond(),FakeFlow(),loader,torch.device('cpu'),
        num_steps=1,max_sequences=1)
    assert result['pose_sequences'][0].shape==(229,55,6)
    assert result['expr_sequences'][0].shape==(229,100)

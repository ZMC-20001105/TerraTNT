#!/usr/bin/env python3
"""Train V8 DualDecoder: 3-stage training."""
import sys, json, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_terratnt_10s import FASDataset, HISTORY_LEN, FUTURE_LEN
from scripts.train_incremental_models import DualDecoderV8, ade_fde_m

def val_model(model, vl, dev, mode='full', fg=None):
    model.eval()
    a,f,g,n=0,0,0,0
    with torch.no_grad():
        for b in vl:
            h,fd,em,c=b['history'].to(dev),b['future'].to(dev),b['env_map'].to(dev),b['candidates'].to(dev)
            gt=torch.cumsum(fd,dim=1); cp=torch.zeros(h.size(0),2,device=dev)
            with autocast(enabled=True):
                o=model(em,h,c,cp,use_gt_goal=False,mode=mode,force_gate=fg)
            ad,fd2=ade_fde_m(o[0],gt); a+=ad.sum().item(); f+=fd2.sum().item(); n+=h.size(0)
            if len(o)>2 and o[2].numel()>1: g+=o[2].sum().item()
    return a/max(1,n),f/max(1,n),g/max(1,n)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--traj_dir',default='outputs/dataset_experiments/D1_optimal_combo')
    pa.add_argument('--split_file',default='outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json')
    pa.add_argument('--output_dir',default='runs/incremental_models_v8')
    pa.add_argument('--batch_size',type=int,default=32)
    pa.add_argument('--num_epochs',type=int,default=60)
    pa.add_argument('--lr',type=float,default=1e-3)
    pa.add_argument('--resume_v6',type=str,default=None)
    pa.add_argument('--patience',type=int,default=10)
    args=pa.parse_args()

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd=Path(args.output_dir); sd.mkdir(parents=True,exist_ok=True)
    print(f'Device: {dev}',flush=True)

    # Dataset
    ds_kw=dict(history_len=HISTORY_LEN,future_len=FUTURE_LEN,num_candidates=6,
               region='bohemian_forest',env_coverage_km=140.0,coord_scale=1.0)
    tr_ds=FASDataset(args.traj_dir,args.split_file,phase='fas1',**ds_kw)
    va_ds=FASDataset(args.traj_dir,args.split_file,phase='fas1',**ds_kw)
    with open(args.split_file) as f: sp=json.load(f)
    tr_ds.samples_meta=[(str(Path(args.traj_dir)/i['file']),int(i['sample_idx'])) for i in sp['fas1']['train_samples']]
    va_ds.samples_meta=[(str(Path(args.traj_dir)/i['file']),int(i['sample_idx'])) for i in sp['fas1']['val_samples']]
    tl=DataLoader(tr_ds,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    vl=DataLoader(va_ds,batch_size=args.batch_size,shuffle=False,num_workers=1,pin_memory=True)
    print(f'Train:{len(tr_ds)} Val:{len(va_ds)}',flush=True)

    # Model
    model=DualDecoderV8(history_dim=26,hidden_dim=128,env_channels=18,env_feature_dim=128,
                        decoder_hidden_dim=256,future_len=FUTURE_LEN,num_waypoints=10,
                        num_candidates=6,env_coverage_km=140.0).to(dev)
    print(f'V8 params: {sum(p.numel() for p in model.parameters()):,}',flush=True)
    name='V8'
    scaler=GradScaler(); cls_c=nn.CrossEntropyLoss(); bce_logit=nn.BCEWithLogitsLoss()
    NE=args.num_epochs; pat=args.patience

    # Load V6 if provided
    skip_s1=False
    if args.resume_v6:
        ck=torch.load(args.resume_v6,map_location=dev,weights_only=False)
        s=ck.get('model_state_dict',ck); ms={}
        for k,v in s.items():
            if k.startswith('fusion.'): ms[k.replace('fusion.','ga_fusion.')]=v
            elif k.startswith('decoder_lstm.'): ms[k.replace('decoder_lstm.','ga_decoder_lstm.')]=v
            elif k.startswith('output_fc.'): ms[k.replace('output_fc.','ga_output_fc.')]=v
            else: ms[k]=v
        mi,_=model.load_state_dict(ms,strict=False)
        print(f'Loaded V6 (ADE={ck.get("val_ade",0):.0f}m), new layers: {len(mi)}',flush=True)
        skip_s1=True

    s1e=0 if skip_s1 else int(NE*0.4)
    s2e=int(NE*0.3)
    s3e=NE-s1e-s2e
    best=float('inf')

    def _save(va,vf,ep):
        nonlocal best
        if va<best:
            best=va
            torch.save({'model_state_dict':model.state_dict(),'epoch':ep,
                        'val_ade':va,'val_fde':vf,'model_name':name},sd/f'{name}_best.pth')
            return True
        return False

    # === STAGE 1 ===
    if s1e>0:
        print(f'\n=== Stage1: Goal-Aware ({s1e}ep) ===',flush=True)
        opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
        ni=0
        for ep in range(s1e):
            model.train(); tl_sum,nb=0,0
            for b in tl:
                h,fd,em,c,ti=b['history'].to(dev),b['future'].to(dev),b['env_map'].to(dev),b['candidates'].to(dev),b['target_goal_idx'].to(dev)
                gt=torch.cumsum(fd,dim=1); cp=torch.zeros(h.size(0),2,device=dev)
                opt.zero_grad()
                with autocast(enabled=True):
                    tr,gl,wp,_,_=model(em,h,c,cp,target_goal_idx=ti,use_gt_goal=True,mode='goal_aware',force_gate=1.0)
                    lo=F.mse_loss(tr,gt)+cls_c(gl,ti)
                    if wp is not None:
                        wi=torch.tensor(model.waypoint_indices,device=dev,dtype=torch.long)
                        lo+=0.5*F.mse_loss(wp,gt.index_select(1,wi))
                scaler.scale(lo).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update()
                tl_sum+=lo.item(); nb+=1
            va,vf,_=val_model(model,vl,dev,'goal_aware',1.0); sch.step(va)
            imp=' *BEST*' if _save(va,vf,ep) else ''; ni=0 if imp else ni+1
            print(f'  [S1] {ep+1}/{s1e} loss={tl_sum/nb:.4f} ADE={va:.0f}m{imp}',flush=True)
            if ni>=pat: break
        best=float('inf')

    # === STAGE 2 ===
    print(f'\n=== Stage2: Goal-Free ({s2e}ep) ===',flush=True)
    ga_pf=('goal_fc','ga_fusion','wp_','spatial_in','env_local_scale','ga_decoder_lstm','ga_output_fc','seg_proj','goal_classifier','env_encoder','history_encoder')
    for n2,p in model.named_parameters(): p.requires_grad=not any(n2.startswith(x) for x in ga_pf)
    trp=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(trp,lr=args.lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5); ni=0
    for ep in range(s2e):
        model.train(); tl_sum,nb=0,0
        for b in tl:
            h,fd,em,c,ti=b['history'].to(dev),b['future'].to(dev),b['env_map'].to(dev),b['candidates'].to(dev),b['target_goal_idx'].to(dev)
            gt=torch.cumsum(fd,dim=1); cp=torch.zeros(h.size(0),2,device=dev)
            opt.zero_grad()
            with autocast(enabled=True):
                tr,_,_,_,_=model(em,h,c,cp,target_goal_idx=ti,use_gt_goal=True,mode='goal_free')
                lo=F.mse_loss(tr,gt)
            scaler.scale(lo).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trp,5.0); scaler.step(opt); scaler.update()
            tl_sum+=lo.item(); nb+=1
        va,vf,_=val_model(model,vl,dev,'goal_free'); sch.step(va)
        imp=' *BEST*' if _save(va,vf,ep) else ''; ni=0 if imp else ni+1
        print(f'  [S2] {ep+1}/{s2e} loss={tl_sum/nb:.4f} ADE={va:.0f}m{imp}',flush=True)
        if ni>=pat: break
    for p in model.parameters(): p.requires_grad=True
    best=float('inf')

    # === STAGE 3 ===
    print(f'\n=== Stage3: Joint Gate ({s3e}ep) ===',flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr*0.3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5); ni=0
    for ep in range(s3e):
        model.train(); tl_sum,nb,gs=0,0,0
        for b in tl:
            h,fd,em,c,ti=b['history'].to(dev),b['future'].to(dev),b['env_map'].to(dev),b['candidates'].to(dev),b['target_goal_idx'].to(dev)
            gt=torch.cumsum(fd,dim=1); cp=torch.zeros(h.size(0),2,device=dev); B=h.size(0)
            corrupt=(torch.rand(1).item()>0.5)
            if corrupt:
                ci=torch.randn(B,c.size(1),2,device=dev)*20.0; gtgt=torch.zeros(B,1,device=dev)
            else:
                ci=c; gtgt=torch.ones(B,1,device=dev)
            opt.zero_grad()
            with autocast(enabled=True):
                tr,gl,wp,al,al_logit=model(em,h,ci,cp,target_goal_idx=ti,use_gt_goal=(not corrupt),mode='full')
                lo=F.mse_loss(tr,gt)+0.5*bce_logit(al_logit,gtgt)
                if not corrupt:
                    lo+=cls_c(gl,ti)
                    if wp is not None:
                        wi=torch.tensor(model.waypoint_indices,device=dev,dtype=torch.long)
                        lo+=0.5*F.mse_loss(wp,gt.index_select(1,wi))
            scaler.scale(lo).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update()
            tl_sum+=lo.item(); nb+=1; gs+=al.detach().mean().item()
        va,vf,vg=val_model(model,vl,dev,'full'); sch.step(va)
        imp=' *BEST*' if _save(va,vf,ep) else ''; ni=0 if imp else ni+1
        print(f'  [S3] {ep+1}/{s3e} loss={tl_sum/nb:.4f} ADE={va:.0f}m gate={gs/nb:.3f} vgate={vg:.3f}{imp}',flush=True)
        if ni>=pat: break

    print(f'\n✅ V8 training done. Best ADE={best:.0f}m → {sd}/{name}_best.pth',flush=True)

if __name__=='__main__':
    main()

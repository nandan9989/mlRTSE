import os, json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

case_names = ["Case1_NucOnly", "Case2_NoEMA", "Case3_SurfOnly", "Case4_BothEMAs"]
case2idx = {c: i for i, c in enumerate(case_names)}

LAMBDA_REG     = 1e-3
BATCH_SIZE     = 64
LR             = 1e-3
EPOCHS         = 80
PATIENCE       = 12
NUM_WORKERS    = 4
MAX_GRAD_NORM  = 1.0

cdte_root      = r"C:\Users\nmuthan\Desktop\Newdata\CdTe"
testdata_dir   = os.path.join(cdte_root, "traindata")
valdata_dir    = os.path.join(cdte_root, "valdata")

reg_indices = [0,1,2,4,5]  # t_SurfEMA, t_Bulk, t_NucEMA, v_SurfEMA, v_NucEMA
reg_names = ["t_SurfEMA", "t_Bulk", "t_NucEMA", "v_SurfEMA", "v_NucEMA"]

class EllipDataset(Dataset):
    def __init__(self, folders):
        self.files = []
        for fld in folders:
            fps = list(Path(fld).glob("*.csv"))
            self.files += fps
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        csv_fp = self.files[idx]
        jsn_fp = csv_fp.with_suffix(".json")
        arr = np.loadtxt(csv_fp, delimiter=",", skiprows=1, usecols=(1,2,3))
        spec = torch.from_numpy(arr.T.astype(np.float32))
        meta = json.load(open(jsn_fp))
        lbl  = case2idx[meta["case"]]
        th = meta["thickness"]
        vf = meta["void_fraction"]
        reg_full = np.array([
            th.get("SurfaceEMA",0.0),
            th.get("Bulk",      0.0),
            th.get("NucEMA",    0.0),
            th.get("Oxide",     0.0),
            vf.get("SurfaceEMA",0.0),
            vf.get("NucEMA",    0.0),
            meta["AOI_deg"]
        ], dtype=np.float32)
        reg = reg_full[reg_indices]
        return spec, lbl, torch.from_numpy(reg)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.se(x)
        return x * w.unsqueeze(-1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            SEBlock(channels)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.net(x))

class ImprovedCNN1D(nn.Module):
    def __init__(self, num_classes, num_reg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(3,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ResidualBlock(32),
            ResidualBlock(32),

            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ResidualBlock(128),
            ResidualBlock(128),

            nn.Conv1d(128,256,3,padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            nn.Conv1d(256,512,3,padding=1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ResidualBlock(512),
            ResidualBlock(512),

            nn.AdaptiveAvgPool1d(1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(512,1024), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(512,1024), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(1024, num_reg)
        )
    def forward(self, x):
        feat = self.layers(x).squeeze(-1)
        return self.cls_head(feat), self.reg_head(feat)

class EarlyStop:
    def __init__(self, patience=PATIENCE, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter = float("inf"), 0
    def __call__(self, loss):
        if loss + self.min_delta < self.best:
            self.best, self.counter = loss, 0
            return False
        self.counter += 1
        return self.counter >= self.patience

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}\n")

    train_dirs = [os.path.join(cdte_root,c) for c in case_names] + [testdata_dir]
    val_dirs   = [valdata_dir]

    train_ds = EllipDataset(train_dirs)
    val_ds   = EllipDataset(val_dirs)
    print(f" • train size = {len(train_ds)}")
    print(f" • val   size = {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model     = ImprovedCNN1D(len(case_names), len(reg_indices)).to(device)
    crit_c    = nn.CrossEntropyLoss(label_smoothing=0.1) if hasattr(nn, "CrossEntropyLoss") and "label_smoothing" in nn.CrossEntropyLoss.__init__.__code__.co_varnames else nn.CrossEntropyLoss()
    reg_weights = torch.tensor([1, 5, 1, 1, 1], dtype=torch.float32).to(device)
    def crit_r(pred, target):
        return torch.mean(reg_weights * torch.abs(pred - target))

    optim     = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    stopper   = EarlyStop()
    best_val  = float("inf")

    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        model.train()
        running_c = running_r = correct = total = 0
        loop = tqdm(train_loader, desc=" Train", ncols=80)
        for spec, lbl, reg in loop:
            spec, lbl, reg = spec.to(device), lbl.to(device), reg.to(device)
            optim.zero_grad()
            o_c, o_r   = model(spec)
            loss_c     = crit_c(o_c, lbl)
            loss_r     = crit_r(o_r, reg)
            loss       = loss_c + LAMBDA_REG * loss_r
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optim.step()
            b           = spec.size(0)
            running_c  += loss_c.item() * b
            running_r  += loss_r.item() * b
            correct    += (o_c.argmax(1) == lbl).sum().item()
            total      += b
            loop.set_postfix(
                cls = running_c/total,
                reg = running_r/total,
                acc = 100*correct/total
            )

        # Validate
        model.eval()
        val_c = val_r = val_corr = vtotal = 0
        all_lbl, all_pred, all_t, all_p = [], [], [], []
        with torch.no_grad():
            loop = tqdm(val_loader, desc=" Val  ", ncols=80)
            for spec, lbl, reg in loop:
                spec, lbl, reg = spec.to(device), lbl.to(device), reg.to(device)
                o_c, o_r   = model(spec)
                loss_c     = crit_c(o_c, lbl)
                loss_r     = crit_r(o_r, reg)
                b          = spec.size(0)
                val_c     += loss_c.item() * b
                val_r     += loss_r.item() * b
                val_corr  += (o_c.argmax(1) == lbl).sum().item()
                vtotal    += b

                all_lbl   .extend(lbl.cpu().numpy())
                all_pred  .extend(o_c.argmax(1).cpu().numpy())
                all_t     .extend(reg.cpu().numpy())
                all_p     .extend(o_r.cpu().numpy())
                loop.set_postfix(
                    cls = val_c/vtotal,
                    reg = val_r/vtotal,
                    acc = 100*val_corr/vtotal
                )

        val_comb = val_c + LAMBDA_REG * val_r
        print(f"▶ Val combined loss: {val_comb/vtotal:.4f}")
        scheduler.step(val_comb)
        current_lr = optim.param_groups[0]['lr']
        print(f"▶ Learning rate: {current_lr:.2e}")

        if val_comb < best_val:
            best_val = val_comb
            torch.save(model.state_dict(), "best_model_ultradeep.pt")
            print("💾 Saved best_model_ultradeep.pt")
        if stopper(val_comb):
            print("⏹ Early stopping")
            break

    # Final evaluation
    print("\n▶ Final evaluation on validation set:")
    model.load_state_dict(torch.load("best_model_ultradeep.pt", map_location=device))
    print(classification_report(all_lbl, all_pred, target_names=case_names, digits=4))
    all_t, all_p = np.array(all_t), np.array(all_p)
    maes = np.mean(np.abs(all_t - all_p), axis=0)
    for n, m in zip(reg_names, maes):
        print(f"MAE {n}: {m:.4f}")

if __name__ == "__main__":
    main()

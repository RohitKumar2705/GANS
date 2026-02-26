import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image


def run(data, G, D, epoch, z_val_):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flip_label = 0.1
    smooth_label = 0.3

    optimizer_d = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_g = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

    loss = nn.BCEWithLogitsLoss()

    for i, X in enumerate(data):

        k = 1
        x = X.float().to(device)
        m = len(x)

        # real labels (smoothed)
        y_ = torch.Tensor(m).uniform_(1 - smooth_label, 1).to(device)
        y = y_.float()

        # fake labels
        w_ = torch.Tensor(m).uniform_(0, smooth_label).to(device)
        w = w_.float()

        # =====================
        # Train Discriminator
        # =====================
        for j in range(k):

            D.zero_grad()

            z1 = torch.randn(m, 100, device=device)
            z_d = z1.float()

            if torch.rand(1).item() > flip_label:
                loss_d_x = loss(D(x), y)
            else:
                loss_d_x = loss(D(x), w)

            # generate fake images (no grad for G)
            with torch.no_grad():
                z_d_2 = G(z_d)

            if torch.rand(1).item() > flip_label:
                loss_d_z = loss(D(z_d_2), w)
            else:
                loss_d_z = loss(D(z_d_2), y)

            loss_d = loss_d_x + loss_d_z
            loss_d.backward()
            optimizer_d.step()

        # =====================
        # Train Generator
        # =====================
        G.zero_grad()

        z2 = torch.randn(m, 100, device=device)
        z_g = z2.float()

        if torch.rand(1).item() > flip_label:
            loss_g = loss(D(G(z_g)), y)
        else:
            loss_g = loss(D(G(z_g)), w)

        loss_g.backward()
        optimizer_g.step()

        # ✅ FIXED LINE 59
        if i % 50 == 0:
            print(
                'loss_d: {} loss_d_real {} loss_d_fake {} - loss_g: {}'.format(
                    loss_d.item(),
                    loss_d_x.item(),
                    loss_d_z.item(),
                    loss_g.item()
                )
            )

    # =====================
    # SAVE SAMPLES
    # =====================
    z_val = z_val_.float().to(device)

    G.eval()
    with torch.no_grad():
        sample = G(z_val)

    # ✅ FIXED LINE 65
    print("saving sample...", sample.size())

    G.train()

    filename = 'results/results-{}.jpeg'.format(epoch + 1)

    save_image(
        sample,
        filename,
        nrow=sample.shape[0] // 8,   # FIXED division
        normalize=True
    )


from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

if torch.cuda.is_available():
    urzadzenie = torch.device("cuda")
elif torch.backends.mps.is_available():
    urzadzenie = torch.device("mps")
else:
    urzadzenie = torch.device("cpu")
print(f"Urządzenie: {urzadzenie}")

przeksztalcenie_testowe = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])
przeksztalcenie_treningowe = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    przeksztalcenie_testowe
])

zbior_treningowy = datasets.CIFAR10(
    root="./dane", train=True, download=True, transform=przeksztalcenie_treningowe
)
zbior_walidacyjny = copy(zbior_treningowy)
zbior_walidacyjny.transform = przeksztalcenie_testowe
zbior_testowy = datasets.CIFAR10(
    root="./dane", train=False, download=True, transform=przeksztalcenie_testowe
)

indeksy = torch.randperm(
    len(zbior_treningowy), generator=torch.Generator().manual_seed(42)
).tolist()
liczba_walidacyjna = 5000
indeksy_treningowe = indeksy[liczba_walidacyjna:]
indeksy_walidacyjne = indeksy[:liczba_walidacyjna]
ladowacz_treningowy = DataLoader(
    Subset(zbior_treningowy, indeksy_treningowe),
    batch_size=128,
    shuffle=True
)
ladowacz_walidacyjny = DataLoader(
    Subset(zbior_walidacyjny, indeksy_walidacyjne),
    batch_size=256
)
ladowacz_testowy = DataLoader(zbior_testowy, batch_size=256)

print(f"Liczba obrazów treningowych: {len(ladowacz_treningowy.dataset)}")
print(f"Liczba obrazów walidacyjnych: {liczba_walidacyjna}")
print(f"Liczba obrazów testowych: {len(zbior_testowy)}")


class WarstwaHighway(nn.Module):
    def __init__(self, kanaly):
        super().__init__()
        self.przeksztalcenie = nn.Sequential(
            nn.Conv2d(kanaly, kanaly, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(kanaly),
            nn.ReLU(inplace=True)
        )
        self.brama = nn.Conv2d(kanaly, kanaly, kernel_size=1)
        nn.init.kaiming_normal_(
            self.przeksztalcenie[0].weight,
            mode="fan_out",
            nonlinearity="relu"
        )
        nn.init.constant_(self.brama.bias, -2.0)

    def forward(self, x):
        h = self.przeksztalcenie(x)
        t = torch.sigmoid(self.brama(x))
        return h * t + x * (1 - t)


class SiecHighway(nn.Module):
    def __init__(self):
        super().__init__()
        self.wejscie = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.etap1 = nn.Sequential(*[WarstwaHighway(24) for _ in range(4)])
        self.przejscie1 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.etap2 = nn.Sequential(*[WarstwaHighway(48) for _ in range(4)])
        self.przejscie2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.etap3 = nn.Sequential(*[WarstwaHighway(96) for _ in range(3)])
        self.klasyfikator = nn.Linear(96, 10)

    def forward(self, x):
        x = self.wejscie(x)
        x = self.etap1(x)
        x = self.przejscie1(x)
        x = self.etap2(x)
        x = self.przejscie2(x)
        x = self.etap3(x)
        x = x.mean(dim=(2, 3))
        return self.klasyfikator(x)


def ocen_model(ladowacz):
    model.eval()
    poprawne = 0
    wszystkie = 0
    suma_strat = 0.0

    with torch.inference_mode():
        for obrazy, etykiety in ladowacz:
            wyniki = model(obrazy.to(urzadzenie))
            etykiety = etykiety.to(urzadzenie)
            suma_strat = suma_strat + funkcja_straty(wyniki, etykiety) * etykiety.size(0)
            poprawne = poprawne + (wyniki.argmax(dim=1) == etykiety).sum()
            wszystkie += etykiety.size(0)

    return suma_strat.item() / wszystkie, 100 * poprawne.item() / wszystkie


torch.manual_seed(42)
model = SiecHighway().to(urzadzenie)
funkcja_straty = nn.CrossEntropyLoss()
optymalizator = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)
liczba_epok = 40
harmonogram = optim.lr_scheduler.CosineAnnealingLR(
    optymalizator,
    T_max=liczba_epok
)

print(f"Liczba parametrów: {sum(p.numel() for p in model.parameters()):,}")
najlepsza_dokladnosc = -1.0
najmniejsza_strata_walidacyjna = float("inf")
najlepsza_epoka = 0
najlepsze_wagi = None

for epoka in range(liczba_epok):
    model.train()
    suma_strat = 0.0
    liczba_obrazow = 0

    for obrazy, etykiety in ladowacz_treningowy:
        obrazy = obrazy.to(urzadzenie)
        etykiety = etykiety.to(urzadzenie)

        optymalizator.zero_grad(set_to_none=True)
        wyniki = model(obrazy)
        strata = funkcja_straty(wyniki, etykiety)
        strata.backward()
        optymalizator.step()
        suma_strat = suma_strat + strata.detach() * etykiety.size(0)
        liczba_obrazow += etykiety.size(0)

    harmonogram.step()
    srednia_strata = suma_strat.item() / liczba_obrazow
    strata_walidacyjna, dokladnosc_walidacyjna = ocen_model(ladowacz_walidacyjny)
    if (dokladnosc_walidacyjna, -strata_walidacyjna) > (
        najlepsza_dokladnosc, -najmniejsza_strata_walidacyjna
    ):
        najlepsza_dokladnosc = dokladnosc_walidacyjna
        najmniejsza_strata_walidacyjna = strata_walidacyjna
        najlepsza_epoka = epoka + 1
        najlepsze_wagi = {
            nazwa: wartosc.detach().clone()
            for nazwa, wartosc in model.state_dict().items()
        }

    print(
        f"Epoka {epoka + 1}/{liczba_epok}, "
        f"strata treningowa: {srednia_strata:.4f}, "
        f"strata walidacyjna: {strata_walidacyjna:.4f}, "
        f"dokładność walidacyjna: {dokladnosc_walidacyjna:.2f}%"
    )

model.load_state_dict(najlepsze_wagi)
torch.save(
    {nazwa: wartosc.cpu() for nazwa, wartosc in najlepsze_wagi.items()},
    "model_highway11_validated.pt"
)
strata_testowa, dokladnosc_testowa = ocen_model(ladowacz_testowy)
print(f"Najlepsza epoka walidacyjna: {najlepsza_epoka}")
print(f"Najlepsza dokładność walidacyjna: {najlepsza_dokladnosc:.2f}%")
print(f"Strata testowa: {strata_testowa:.4f}")
print(f"Dokładność testowa: {dokladnosc_testowa:.2f}%")

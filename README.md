# Racetrack — Monte Carlo Control

Cài đặt bài tập 5.12 (Sutton & Barto) với 4 thuật toán Monte Carlo:
- On-policy ε-soft (First-visit MC)
- Off-policy WIS (Weighted Importance Sampling)
- Off-policy OIS (Ordinary Importance Sampling)
- Per-decision Importance Sampling

## Cấu trúc project

```
racetrack_mc/
├── config.py       # Tham số toàn cục
├── track.py        # Môi trường Racetrack
├── agents.py       # 4 thuật toán MC
├── train.py        # Vòng lặp training
├── plot.py         # Visualization
├── main.py         # Entry point
└── requirements.txt
```

## Chạy local (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Chạy trên Kaggle

Dùng file `notebook.ipynb` — import trực tiếp các module và chạy từng cell.
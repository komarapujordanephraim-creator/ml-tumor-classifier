import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.datasets import load_breast_cancer
from train import build_models


class TestModels(unittest.TestCase):
    def test_models_beat_baseline(self):
        X, y = load_breast_cancer(return_X_y=True)
        for name, m in build_models().items():
            m.fit(X[:400], y[:400])
            acc = (m.predict(X[400:]) == y[400:]).mean()
            self.assertGreater(acc, 0.85, name)


if __name__ == "__main__":
    unittest.main()

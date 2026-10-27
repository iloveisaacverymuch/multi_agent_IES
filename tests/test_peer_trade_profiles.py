from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_peer_trade_profiles import signed_trade_by_counterparty


def test_signed_trade_by_counterparty_uses_buyer_positive_seller_negative():
    records = [
        [1, 2, 4.0, 0.5],
        [3, 1, 1.5, 0.6],
        [1, 2, 2.0, 0.7],
    ]

    assert signed_trade_by_counterparty(records, 1, 2) == 6.0
    assert signed_trade_by_counterparty(records, 2, 1) == -6.0
    assert signed_trade_by_counterparty(records, 1, 3) == -1.5
    assert signed_trade_by_counterparty(records, 3, 1) == 1.5

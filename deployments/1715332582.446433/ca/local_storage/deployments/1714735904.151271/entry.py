
from tg.common.delivery.delivery import EntryPoint
from pathlib import Path

entry_point = EntryPoint("Mnist_container", "1", Path(__file__).parent/"entry.pkl")
entry_point.run()


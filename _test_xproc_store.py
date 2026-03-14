"""Quick cross-process shared-memory test."""
import sys, time, subprocess
sys.path.insert(0, ".")
from pipeline.nodus_loss_store import NodusLossStore

store = NodusLossStore.get_global()
store.clear()
store.record("test/xproc", loss=0.42, aux=0.0, round_id=1, ts=time.time())
print("Parent wrote loss=0.42 to test/xproc")

child_code = (
    "import sys; sys.path.insert(0,'.'); "
    "from pipeline.nodus_loss_store import NodusLossStore; "
    "s=NodusLossStore.get_global(); "
    "rec=s.latest('test/xproc'); "
    "print(f'CHILD READ: {rec}')"
)
result = subprocess.run([sys.executable, "-c", child_code],
                        capture_output=True, text=True)
print("CHILD STDOUT:", repr(result.stdout))
print("CHILD STDERR:", repr(result.stderr))
print("CHILD RC:", result.returncode)
store.clear()
print("PASS" if "0.41999" in result.stdout else "FAIL")

"""Quick cross-process smoke test for the runtime control shared store."""
import subprocess
import sys
import time

sys.path.insert(0, ".")

from pipeline.nodus_loss_store import NodusRuntimeControlStore


store = NodusRuntimeControlStore.get_global()
store.clear()
store.begin_service("parent", ts=time.time())
store.set_exit_requested(True, reason="parent_exit", ts=time.time())
state = store.get_state()
print("PARENT STATE:", state)

child_code = (
    "import sys; sys.path.insert(0,'.'); "
    "from pipeline.nodus_loss_store import NodusRuntimeControlStore; "
    "s=NodusRuntimeControlStore.get_global(); "
    "st=s.get_state(); "
    "print(st)"
)
result = subprocess.run([sys.executable, "-c", child_code], capture_output=True, text=True)
print("CHILD STDOUT:", repr(result.stdout))
print("CHILD STDERR:", repr(result.stderr))
print("CHILD RC:", result.returncode)

store.clear()

ok = "active_service_count=1" in result.stdout and "exit_requested=True" in result.stdout
print("PASS" if ok else "FAIL")

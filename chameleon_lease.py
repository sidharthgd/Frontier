from chi import context, lease
from datetime import datetime, timedelta, timezone

# Reserve resources via python-chi
# Requires: pip install python-chi

context.use_device_auth()
context.use_site("CHI@TACC")
context.use_project("CHI-261731")  # replace with your Chameleon project name

my_lease = lease.Lease(
    "my-reservation",
    duration=timedelta(days=3, hours=8),
)
my_lease.add_node_reservation(amount=1, node_type="compute_zen3")
my_lease.submit(idempotent=True)
print(f"Lease created: {my_lease.id}")

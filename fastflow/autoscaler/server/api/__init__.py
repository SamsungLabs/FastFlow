from fastflow.autoscaler.server.api import heartbeat
from fastflow.autoscaler.server.api import instance_infos

routes = {
    "/api/heartbeat": heartbeat.route,
    "/api/instance_infos": instance_infos.route
}

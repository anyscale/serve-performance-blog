from locust import HttpUser, task, constant, events
import random


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--token", type=str, default="", help="Auth bearer token")
    parser.add_argument(
        "--route_prefix", type=str, default="/", help="Route prefix for requests"
    )


class DLRMUser(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        token = self.environment.parsed_options.token
        if token:
            self.client.headers.update({"Authorization": f"Bearer {token}"})

    @task
    def get_recommendations(self):
        route_prefix = self.environment.parsed_options.route_prefix
        user_id = random.randint(1, 1000)
        with self.client.get(
            route_prefix,
            params={"user_id": user_id},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    if "scores" in json_response:
                        response.success()
                    else:
                        response.failure("Response missing 'scores' field")
                except Exception as e:
                    response.failure(f"Failed to parse JSON: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")

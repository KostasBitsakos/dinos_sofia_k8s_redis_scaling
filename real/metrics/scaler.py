from kubernetes import client, config
import metricsnew

def scale_deployment_based_on_usage(namespace="default", deployment_name="redis-autoscale-cluster", cpu_threshold_increase=0.10, cpu_threshold_decrease=0.05, memory_threshold_increase=0.50, memory_threshold_decrease=0.3):
    # Fetch average CPU and memory usage
    avg_cpu_usage, avg_memory_usage = metricsnew.calculate_average_usage(namespace)

    # Load the kubeconfig file
    config.load_kube_config()

    # Create an instance of the API class
    api_instance = client.AppsV1Api()
    # Get the current state of the deployment
    deployment = api_instance.read_namespaced_deployment(deployment_name, namespace)
    current_replicas = deployment.spec.replicas

    # Decide on scaling action based on thresholds
    if avg_cpu_usage > cpu_threshold_increase or avg_memory_usage > memory_threshold_increase:
        # Scale up
        additional_replicas = 1
    elif avg_cpu_usage < cpu_threshold_decrease and avg_memory_usage < memory_threshold_decrease:
        # Scale down, but ensure replicas don't go below 1
        additional_replicas = -1 if current_replicas > 1 else 0
    else:
        # No action needed
        additional_replicas = 0

    if additional_replicas != 0:
        # Calculate the new desired number of replicas
        new_replicas = current_replicas + additional_replicas

        # Update the deployment
        deployment.spec.replicas = new_replicas
        api_instance.patch_namespaced_deployment(deployment_name, namespace, deployment)
        print(f"Deployment {deployment_name} scaled to {new_replicas} replicas.")
    else:
        print("No scaling action needed.")
def main():
    scale_deployment_based_on_usage()

if __name__ == '__main__':
    main()

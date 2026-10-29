from kubernetes import client, config

def scale_deployment(namespace, deployment_name, additional_replicas):
    # Load the kubeconfig file
    config.load_kube_config()

    # Create an instance of the API class
    api_instance = client.AppsV1Api()
    # Get the current state of the deployment
    deployment = api_instance.read_namespaced_deployment(deployment_name, namespace)
    current_replicas = deployment.spec.replicas

    # Calculate the new desired number of replicas
    new_replicas = current_replicas + additional_replicas

    # Update the deployment
    deployment.spec.replicas = new_replicas
    api_instance.patch_namespaced_deployment(deployment_name, namespace, deployment)
    print(f"Deployment {deployment_name} scaled to {new_replicas} replicas.")

# Example usage
namespace = "default"
deployment_name = "redis-autoscale-cluster"
additional_replicas = 1  # Increase by one
scale_deployment(namespace, deployment_name, additional_replicas)


from kubernetes import client, config

def list_pod_metrics(namespace='default'):
    # Load the kubeconfig file
    config.load_kube_config()

    # Initialize the Metrics API
    api_instance = client.CustomObjectsApi()
    group = 'metrics.k8s.io'
    version = 'v1beta1'
    namespace = namespace
    plural = 'pods'

    try:
        # Fetch the metrics for all pods in the specified namespace
        api_response = api_instance.list_namespaced_custom_object(group, version, namespace, plural)
        print("CPU and Memory usage for pods in namespace '{}':\n".format(namespace))
        for pod_metrics in api_response['items']:
            pod_name = pod_metrics['metadata']['name']
            cpu_usage = pod_metrics['containers'][0]['usage']['cpu']
            memory_usage = pod_metrics['containers'][0]['usage']['memory']
            print("Pod: {}, CPU: {}, Memory: {}".format(pod_name, cpu_usage, memory_usage))
    except client.exceptions.ApiException as e:
        print("Exception when calling CustomObjectsApi->list_namespaced_custom_object: %s\n" % e)

# List metrics for pods in the default namespace
list_pod_metrics('default')


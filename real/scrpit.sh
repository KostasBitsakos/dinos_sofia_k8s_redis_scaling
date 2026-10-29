#!/bin/bash

# Exit script on error
set -e

# Delete existing deployments
kubectl delete -f redis-autoscale-cluster-deployment.yaml
sleep 5
kubectl delete -f redis-cluster-coordinator-deployment.yaml

# Wait for resources to be fully deleted
sleep 20

# Recreate deployments
kubectl create -f redis-cluster-coordinator-deployment.yaml
sleep 20
kubectl create -f redis-autoscale-cluster-deployment.yaml

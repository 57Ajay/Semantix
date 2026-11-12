# Kubernetes notes

- Pod: one or more containers that share a network namespace. Pods are disposable.
- Deployment: keeps N replicas of a pod template running, handles rolling updates.
- Service: stable name and address in front of a changing set of pods.
- ConfigMap and Secret: configuration injected as environment variables or files.

## Commands I keep looking up

```
kubectl get pods -n billing -o wide
kubectl describe pod <name>              # events: image pull errors, OOMKilled
kubectl logs <pod> -c <container> --previous
kubectl rollout status deployment/api
kubectl rollout undo deployment/api      # back to the previous version
kubectl port-forward svc/api 8080:80
```

CrashLoopBackOff means the container keeps exiting; check `logs --previous`.

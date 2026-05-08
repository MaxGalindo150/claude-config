---
name: kubernetes-patterns
description: Kubernetes production patterns — Deployments, Services, Ingress, ConfigMaps, Secrets, RBAC, HPA, health probes, resource limits, and Helm chart structure for containerized Python/Node services.
origin: local
---

# Kubernetes Patterns

Production-grade Kubernetes patterns for deploying and operating containerized services.

## When to Activate

- Writing or reviewing Kubernetes manifests
- Setting up Deployments, Services, Ingress
- Configuring HPA (horizontal pod autoscaling)
- Designing RBAC and namespace isolation
- Troubleshooting pod crashes, OOMKilled, or CrashLoopBackOff
- Structuring Helm charts

## Core Resource Patterns

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
  namespace: production
  labels:
    app: my-service
    version: "1.0.0"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-service
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0        # Zero-downtime: never remove before adding
      maxSurge: 1
  template:
    metadata:
      labels:
        app: my-service
    spec:
      serviceAccountName: sa-my-service  # Dedicated SA, not default
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
        - name: my-service
          image: registry/my-service:abc123  # Always use digest or commit SHA
          ports:
            - containerPort: 8080
          env:
            - name: ENV
              value: production
            - name: DB_URL
              valueFrom:
                secretKeyRef:
                  name: my-service-secrets
                  key: db-url
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            failureThreshold: 30   # 30 * 10s = 5 min for slow startups
            periodSeconds: 10
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
  namespace: production
spec:
  selector:
    app: my-service
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP    # Always ClusterIP internally; expose via Ingress
```

### Ingress (nginx)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-service
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
      secretName: api-example-com-tls
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-service
                port:
                  number: 80
```

## Configuration

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-service-config
  namespace: production
data:
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  ALLOWED_ORIGINS: "https://app.example.com"
```

### Secret (base64-encoded)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-service-secrets
  namespace: production
type: Opaque
stringData:           # Use stringData — kubectl encodes automatically
  db-url: "postgresql+asyncpg://user:pass@host/db"
  secret-key: "super-secret"
```

> **Never commit Secrets to git.** Use Sealed Secrets, External Secrets Operator, or Vault. In CI/CD, inject via `kubectl create secret` from vault/CI env vars.

### External Secrets Operator (recommended)

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: my-service-secrets
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: gcp-secret-store    # or aws-secret-store, vault-store
    kind: ClusterSecretStore
  target:
    name: my-service-secrets
  data:
    - secretKey: db-url
      remoteRef:
        key: my-service-db-url
```

## Autoscaling

### HorizontalPodAutoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-service
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### KEDA (event-driven autoscaling)

```yaml
# Scale based on queue depth (Cloud Tasks, SQS, Redis, etc.)
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: my-worker
spec:
  scaleTargetRef:
    name: my-worker
  minReplicaCount: 0     # Scale to zero when no messages
  maxReplicaCount: 10
  triggers:
    - type: gcp-pubsub
      metadata:
        subscriptionName: my-subscription
        value: "5"   # messages per replica
```

## RBAC

### ServiceAccount + Role

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sa-my-service
  namespace: production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: role-my-service
  namespace: production
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "watch", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: rb-my-service
  namespace: production
subjects:
  - kind: ServiceAccount
    name: sa-my-service
    namespace: production
roleRef:
  kind: Role
  name: role-my-service
  apiGroup: rbac.authorization.k8s.io
```

## Namespace Strategy

```
namespaces:
  production    # Live traffic — strict resource quotas, PodDisruptionBudget
  staging       # Pre-prod — mirrors prod config, lower resource limits
  development   # Dev workloads — no quotas, auto-delete after 7 days
```

### ResourceQuota per namespace

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: production-quota
  namespace: production
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    pods: "50"
```

## Helm Chart Structure

```
charts/my-service/
├── Chart.yaml
├── values.yaml          # Defaults
├── values-staging.yaml  # Staging overrides
├── values-prod.yaml     # Production overrides
└── templates/
    ├── _helpers.tpl
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    ├── configmap.yaml
    ├── serviceaccount.yaml
    └── secrets.yaml     # ExternalSecret, not raw Secret
```

```yaml
# values.yaml
replicaCount: 2
image:
  repository: registry/my-service
  tag: latest
  pullPolicy: IfNotPresent
resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## Debugging Runbook

```bash
# Pod not starting
kubectl describe pod <pod-name> -n production
kubectl logs <pod-name> -n production --previous   # Logs from crashed container

# Common issues
kubectl get events -n production --sort-by='.lastTimestamp'

# Exec into running pod
kubectl exec -it <pod-name> -n production -- /bin/sh

# Check resource usage
kubectl top pods -n production
kubectl top nodes

# Port-forward for local debugging
kubectl port-forward svc/my-service 8080:80 -n production
```

### Common Exit Codes

| Code | Cause | Fix |
|------|-------|-----|
| 137 | OOMKilled | Increase `limits.memory` |
| 1 | App crash | Check logs |
| 0 | Clean exit (shouldn't restart) | Check liveness probe config |
| CrashLoopBackOff | Repeated crashes | Check logs, startup probe |

## Production Checklist

- [ ] Resources `requests` and `limits` set on all containers
- [ ] `readinessProbe` and `livenessProbe` configured
- [ ] `startupProbe` for services with slow initialization
- [ ] `maxUnavailable: 0` in rolling update strategy
- [ ] `PodDisruptionBudget` for critical services
- [ ] Secrets via External Secrets, not raw YAML
- [ ] Non-root user in `securityContext`
- [ ] Dedicated `ServiceAccount` per service
- [ ] HPA with sensible min/max replicas
- [ ] `ResourceQuota` per namespace
- [ ] Image tags pinned to commit SHA, not `latest`

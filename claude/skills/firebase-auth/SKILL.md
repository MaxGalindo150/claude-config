---
name: firebase-auth
description: Firebase Authentication patterns — token verification in FastAPI, custom claims for RBAC, multi-tenant auth, ID token validation, service account setup, and frontend Firebase SDK integration.
origin: local
---

# Firebase Authentication Patterns

Patterns for using Firebase Auth as the identity layer in FastAPI backends and React frontends.

## When to Activate

- Verifying Firebase ID tokens in FastAPI
- Adding custom claims (roles, tenant_id) to tokens
- Setting up Firebase Admin SDK
- Implementing multi-tenant authentication
- Debugging auth errors (token expired, wrong audience, etc.)
- Integrating Firebase Auth in React with Zustand

## Backend — FastAPI Integration

### Setup Firebase Admin SDK

```python
# app/core/firebase.py
import firebase_admin
from firebase_admin import auth, credentials
from functools import lru_cache

@lru_cache(maxsize=1)
def get_firebase_app() -> firebase_admin.App:
    """Initialize Firebase Admin once per process."""
    # Option 1: Service account JSON (local dev)
    # cred = credentials.Certificate("path/to/serviceAccount.json")

    # Option 2: Application Default Credentials (Cloud Run, GKE — preferred)
    cred = credentials.ApplicationDefault()

    return firebase_admin.initialize_app(cred, {
        "projectId": settings.FIREBASE_PROJECT_ID,
    })
```

### Token Verification Dependency

```python
from firebase_admin import auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

bearer = HTTPBearer()

class FirebaseUser(BaseModel):
    uid: str
    email: str | None = None
    email_verified: bool = False
    tenant_id: str | None = None  # Custom claim
    role: str = "member"          # Custom claim

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> FirebaseUser:
    token = credentials.credentials
    try:
        decoded = auth.verify_id_token(token, app=get_firebase_app())
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token expired")
    except auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="Token revoked")
    except auth.InvalidIdTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    return FirebaseUser(
        uid=decoded["uid"],
        email=decoded.get("email"),
        email_verified=decoded.get("email_verified", False),
        tenant_id=decoded.get("tenant_id"),  # Custom claim
        role=decoded.get("role", "member"),   # Custom claim
    )

# Optional: require verified email
async def require_verified_email(
    user: FirebaseUser = Depends(get_current_user),
) -> FirebaseUser:
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Email not verified")
    return user
```

### Route Usage

```python
@router.get("/me")
async def get_me(user: FirebaseUser = Depends(get_current_user)):
    return {"uid": user.uid, "tenant": user.tenant_id}

@router.post("/conversations")
async def create_conversation(
    body: ConversationCreate,
    user: FirebaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_tenant_db),
):
    # user.tenant_id comes from Firebase custom claims
    ...
```

## Custom Claims

Custom claims persist in the Firebase ID token — no DB lookup on every request.

### Set Claims (Server-side, Admin SDK)

```python
from firebase_admin import auth

async def set_tenant_claims(uid: str, tenant_id: str, role: str = "member") -> None:
    """Set after user joins a tenant. Token refreshes on next sign-in."""
    auth.set_custom_user_claims(uid, {
        "tenant_id": tenant_id,
        "role": role,
    })

# User must refresh their token to get new claims
# Frontend: await firebase.auth().currentUser.getIdToken(true)  // force refresh
```

### Claims Design

```python
# Keep claims minimal — they're in every token
# Good: scalar values for fast auth decisions
GOOD_CLAIMS = {
    "tenant_id": "acme-corp",    # Which tenant
    "role": "admin",             # Role in that tenant
}

# Bad: large objects, arrays, nested structures
BAD_CLAIMS = {
    "permissions": ["read", "write", "delete", "admin", ...],  # Too large
    "user_data": {"name": "...", "avatar": "..."},             # Not needed in token
}
```

### Invalidate Claims / Force Logout

```python
async def revoke_user_tokens(uid: str) -> None:
    """Force all sessions to expire immediately."""
    auth.revoke_refresh_tokens(uid)

# In verify_id_token, check revocation:
decoded = auth.verify_id_token(token, check_revoked=True, app=get_firebase_app())
```

## Multi-Tenant Auth

When a user can belong to multiple tenants, don't store `tenant_id` in claims. Instead, validate from the request:

```python
# Option A: tenant_id from subdomain/path + validated against DB membership
async def get_current_user_for_tenant(
    tenant_id: str,  # From path param or subdomain middleware
    user: FirebaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[FirebaseUser, TenantMembership]:
    membership = await get_membership(db, tenant_id=tenant_id, user_id=user.uid)
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this tenant")
    return user, membership
```

## Frontend — React Integration

### Firebase Config

```typescript
// src/lib/firebase.ts
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const app = initializeApp({
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
});

export const auth = getAuth(app);
```

### Auth Store (Zustand)

```typescript
// src/store/auth.ts
import { create } from "zustand";
import {
  User,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
  GoogleAuthProvider,
  signInWithPopup,
} from "firebase/auth";
import { auth } from "@/lib/firebase";

interface AuthState {
  user: User | null;
  loading: boolean;
  token: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
  refreshToken: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  loading: true,
  token: null,

  signIn: async (email, password) => {
    await signInWithEmailAndPassword(auth, email, password);
  },

  signInWithGoogle: async () => {
    const provider = new GoogleAuthProvider();
    await signInWithPopup(auth, provider);
  },

  signOut: async () => {
    await signOut(auth);
    set({ user: null, token: null });
  },

  refreshToken: async () => {
    const user = auth.currentUser;
    if (user) {
      const token = await user.getIdToken(true); // force refresh
      set({ token });
    }
  },
}));

// Initialize listener
onAuthStateChanged(auth, async (user) => {
  if (user) {
    const token = await user.getIdToken();
    useAuthStore.setState({ user, token, loading: false });
  } else {
    useAuthStore.setState({ user: null, token: null, loading: false });
  }
});
```

### Axios Interceptor

```typescript
// src/lib/api.ts
import axios from "axios";
import { auth } from "./firebase";

const api = axios.create({ baseURL: import.meta.env.VITE_API_URL });

api.interceptors.request.use(async (config) => {
  const user = auth.currentUser;
  if (user) {
    const token = await user.getIdToken(); // Auto-refreshes if expired
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    if (error.response?.status === 401) {
      // Force token refresh and retry once
      const user = auth.currentUser;
      if (user) {
        const token = await user.getIdToken(true);
        error.config.headers.Authorization = `Bearer ${token}`;
        return api.request(error.config);
      }
    }
    return Promise.reject(error);
  }
);

export default api;
```

## Local Development

```bash
# Use Firebase Emulator for local dev — no real Firebase project needed
firebase emulators:start --only auth

# Point SDK to emulator
export FIREBASE_AUTH_EMULATOR_HOST="localhost:9099"

# FastAPI auto-detects FIREBASE_AUTH_EMULATOR_HOST env var
# No code changes needed
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Token expired` | ID token > 1 hour old | Call `getIdToken(true)` on frontend |
| `Wrong audience` | `projectId` mismatch | Verify `FIREBASE_PROJECT_ID` |
| `Certificate fetch failed` | No internet / firewall | Check Cloud Run egress, use Application Default Credentials |
| `Custom claims too large` | Claims > 1000 bytes | Move non-auth data to DB |
| `Revoked token` | `revoke_refresh_tokens` was called | User must re-authenticate |

## Security Checklist

- [ ] `verify_id_token` called server-side — never trust client-provided user IDs
- [ ] `check_revoked=True` on sensitive operations (payments, admin actions)
- [ ] Custom claims are minimal (< 1000 bytes total)
- [ ] Service account key never committed to git — use Application Default Credentials
- [ ] Email verification required for sensitive endpoints (`require_verified_email`)
- [ ] Token not logged or stored — only used for request auth

## Reference Skills

- Multi-tenant auth flow → skill: `multi-tenant-saas`
- FastAPI dependencies → skill: `fastapi-patterns`
- FastAPI security → skill: `fastapi-security`

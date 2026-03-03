"""
Storage Manager - Backend Abstraction Layer
============================================

Provides a clean abstraction layer for persisting processed tensors and metrics,
allowing seamless swapping between in-memory storage (prototype) and Redis
(production deployment) without modifying other modules.

Key Classes:
------------
- StorageManager: Abstract base class defining the interface
- InMemoryStorage: Streamlit session_state backed implementation (default)
- RedisStorage: Redis-backed implementation (production, currently stubbed)

Architecture Benefits:
----------------------
1. **Separation of Concerns**: DSP pipeline doesn't know about storage backend
2. **Easy Testing**: Can mock storage for unit tests
3. **Production Ready**: One-line config change to switch to Redis
4. **Type Safety**: Clear interface contract via abstract base class

Author: Senior AI Software Engineer & BCI Data Architect
Date: 2026-03-02
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import numpy as np


# ============================================================================
# Abstract Base Class - Storage Interface Contract
# ============================================================================

class StorageManager(ABC):
    """
    Abstract base class defining the storage interface contract.

    All storage backends (in-memory, Redis, database, etc.) must implement
    these methods to ensure compatibility with the DSP pipeline and metrics engine.
    """

    @abstractmethod
    def save_tensor(
        self,
        cleaned_tensor: np.ndarray,
        yield_pct: float,
        metadata: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Save processed tensor and associated metrics to storage.

        Parameters:
        -----------
        cleaned_tensor : np.ndarray
            Processed signal from DSP pipeline (shape: [n_samples], float32)
        yield_pct : float
            Signal Yield percentage (0.0-100.0)
        metadata : dict
            Processing metadata (spike_count, variance, latency_ms, etc.)
        timestamp : float, optional
            Unix timestamp (default: current time)
        """
        pass

    @abstractmethod
    def get_yield_history(self, max_count: int = 200) -> List[float]:
        """
        Retrieve yield history for chronic stability tracking.

        Parameters:
        -----------
        max_count : int
            Maximum number of recent yields to return (default: 200)

        Returns:
        --------
        yield_history : List[float]
            List of yield percentages, ordered chronologically (oldest to newest)
        """
        pass

    @abstractmethod
    def get_latest_tensor(self) -> Optional[np.ndarray]:
        """
        Retrieve the most recently processed tensor.

        Returns:
        --------
        tensor : np.ndarray or None
            Latest cleaned tensor, or None if no data stored
        """
        pass

    @abstractmethod
    def get_metadata_history(self, max_count: int = 200) -> List[Dict[str, Any]]:
        """
        Retrieve metadata history for analytics.

        Parameters:
        -----------
        max_count : int
            Maximum number of recent metadata entries to return

        Returns:
        --------
        metadata_history : List[Dict]
            List of metadata dictionaries, ordered chronologically
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all stored data (for new session).
        """
        pass

    @abstractmethod
    def get_epoch_count(self) -> int:
        """
        Get total number of epochs processed in this session.

        Returns:
        --------
        count : int
            Number of chunks processed since session start
        """
        pass


# ============================================================================
# In-Memory Storage - Default for Prototype
# ============================================================================

class InMemoryStorage(StorageManager):
    """
    In-memory storage backend using Python lists/dicts.

    This implementation is backed by Streamlit's session_state in the actual app,
    but can be used standalone for testing. Data persists only during the current
    session and is lost on page reload/restart.

    Ideal for:
    ----------
    - Rapid prototyping
    - Development/testing
    - Single-session demos
    - Low-latency requirements

    Not suitable for:
    -----------------
    - Multi-session persistence
    - Distributed processing
    - Production long-term monitoring
    """

    def __init__(self, max_history: int = 200):
        """
        Initialize empty in-memory storage with bounded memory.

        Parameters:
        -----------
        max_history : int
            Maximum number of epochs to store before truncating oldest (default: 200)
            Prevents memory leak in long-running sessions
        """
        self.max_history = max_history
        self.tensors: List[np.ndarray] = []
        self.yields: List[float] = []
        self.metadata_list: List[Dict[str, Any]] = []
        self.timestamps: List[float] = []

    def save_tensor(
        self,
        cleaned_tensor: np.ndarray,
        yield_pct: float,
        metadata: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Save tensor and metrics to in-memory lists with automatic truncation.

        O(1) append operation if under limit.
        If at max_history, removes oldest entry (O(n) but only happens at boundary).
        """
        if timestamp is None:
            timestamp = time.time()

        # Store all data in synchronized lists
        self.tensors.append(cleaned_tensor.copy())  # Copy to avoid reference issues
        self.yields.append(yield_pct)
        self.metadata_list.append(metadata.copy())
        self.timestamps.append(timestamp)

        # **MEMORY LEAK FIX**: Truncate to max_history to prevent unbounded growth
        if len(self.tensors) > self.max_history:
            self.tensors.pop(0)
            self.yields.pop(0)
            self.metadata_list.pop(0)
            self.timestamps.pop(0)

    def get_yield_history(self, max_count: int = 200) -> List[float]:
        """
        Retrieve recent yield history.

        Returns last N yields (or all if fewer than N stored).
        """
        if len(self.yields) <= max_count:
            return self.yields.copy()
        else:
            return self.yields[-max_count:]

    def get_latest_tensor(self) -> Optional[np.ndarray]:
        """
        Retrieve most recent tensor.

        Returns None if no tensors stored yet.
        """
        if len(self.tensors) == 0:
            return None
        return self.tensors[-1].copy()

    def get_metadata_history(self, max_count: int = 200) -> List[Dict[str, Any]]:
        """
        Retrieve recent metadata history.

        Returns last N metadata entries (or all if fewer than N stored).
        """
        if len(self.metadata_list) <= max_count:
            return [m.copy() for m in self.metadata_list]
        else:
            return [m.copy() for m in self.metadata_list[-max_count:]]

    def reset(self) -> None:
        """
        Clear all in-memory data.

        Called when user starts a new session or clicks "Reset".
        """
        self.tensors.clear()
        self.yields.clear()
        self.metadata_list.clear()
        self.timestamps.clear()

    def get_epoch_count(self) -> int:
        """
        Get number of epochs processed.

        Since we store one entry per epoch, this is just len(yields).
        """
        return len(self.yields)

    def get_timestamps(self) -> List[float]:
        """
        Get all timestamps (useful for time-series plotting).

        Returns:
        --------
        timestamps : List[float]
            Unix timestamps for each stored epoch
        """
        return self.timestamps.copy()


# ============================================================================
# Redis Storage - Production Backend (Stubbed for Future)
# ============================================================================

class RedisStorage(StorageManager):
    """
    Redis-backed storage for production deployment.

    This implementation is STUBBED for now (returns dummy data) but shows
    the integration path for production. To activate:

    1. Install Redis: `pip install redis`
    2. Deploy Redis server (AWS ElastiCache, Redis Cloud, or local)
    3. Uncomment the actual Redis code below
    4. Set STORAGE_BACKEND = "redis" in app.py

    Benefits of Redis:
    ------------------
    - Multi-session persistence (data survives restart)
    - Distributed access (multiple dashboard instances)
    - High-speed in-memory database (microsecond latency)
    - Built-in data expiration (TTL for old epochs)
    - Pub/Sub for real-time updates across clients

    Storage Schema:
    ---------------
    Key Pattern: axoft:session:{session_id}:epoch:{epoch_num}
    Value: JSON serialized dict with {tensor, yield, metadata, timestamp}
    """

    def __init__(self, host: str = "localhost", port: int = 6379, session_id: str = "default"):
        """
        Initialize Redis connection.

        Parameters:
        -----------
        host : str
            Redis server hostname (default: "localhost")
        port : int
            Redis server port (default: 6379)
        session_id : str
            Unique session identifier for data isolation
        """
        self.host = host
        self.port = port
        self.session_id = session_id

        # UNCOMMENT FOR ACTUAL REDIS INTEGRATION:
        # import redis
        # self.client = redis.Redis(host=host, port=port, decode_responses=False)
        # self.client.ping()  # Verify connection

        # For now, fallback to in-memory (stub)
        print(f"[RedisStorage] STUB MODE: Redis backend not implemented yet.")
        print(f"[RedisStorage] Would connect to redis://{host}:{port} with session_id={session_id}")
        self._stub_storage = InMemoryStorage()

    def save_tensor(
        self,
        cleaned_tensor: np.ndarray,
        yield_pct: float,
        metadata: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Save tensor to Redis.

        STUB: Currently delegates to in-memory storage.

        Production implementation would:
        1. Serialize tensor to bytes (msgpack or pickle)
        2. Create JSON payload with {tensor_bytes, yield, metadata, timestamp}
        3. Store in Redis with key: axoft:session:{session_id}:epoch:{epoch_num}
        4. Set TTL (e.g., 24 hours) to auto-expire old data
        """
        # STUB: Use in-memory fallback
        self._stub_storage.save_tensor(cleaned_tensor, yield_pct, metadata, timestamp)

        # UNCOMMENT FOR ACTUAL REDIS:
        # import msgpack
        # epoch_num = self.get_epoch_count() + 1
        # key = f"axoft:session:{self.session_id}:epoch:{epoch_num}"
        #
        # # Serialize data
        # tensor_bytes = msgpack.packb(cleaned_tensor.tobytes())
        # payload = {
        #     "tensor": tensor_bytes,
        #     "tensor_shape": cleaned_tensor.shape,
        #     "tensor_dtype": str(cleaned_tensor.dtype),
        #     "yield": yield_pct,
        #     "metadata": metadata,
        #     "timestamp": timestamp or time.time()
        # }
        #
        # # Store in Redis with 24-hour TTL
        # self.client.setex(key, 86400, msgpack.packb(payload))

    def get_yield_history(self, max_count: int = 200) -> List[float]:
        """Retrieve yield history from Redis (STUB)."""
        return self._stub_storage.get_yield_history(max_count)

    def get_latest_tensor(self) -> Optional[np.ndarray]:
        """Retrieve latest tensor from Redis (STUB)."""
        return self._stub_storage.get_latest_tensor()

    def get_metadata_history(self, max_count: int = 200) -> List[Dict[str, Any]]:
        """Retrieve metadata history from Redis (STUB)."""
        return self._stub_storage.get_metadata_history(max_count)

    def reset(self) -> None:
        """Clear all session data from Redis (STUB)."""
        self._stub_storage.reset()

        # UNCOMMENT FOR ACTUAL REDIS:
        # # Delete all keys matching session pattern
        # pattern = f"axoft:session:{self.session_id}:epoch:*"
        # for key in self.client.scan_iter(match=pattern):
        #     self.client.delete(key)

    def get_epoch_count(self) -> int:
        """Get epoch count from Redis (STUB)."""
        return self._stub_storage.get_epoch_count()


# ============================================================================
# Factory Function - Create Storage Backend
# ============================================================================

def create_storage(backend: str = "in_memory", **kwargs) -> StorageManager:
    """
    Factory function to create appropriate storage backend.

    This is the ONLY function the rest of the codebase needs to call.
    Changing the backend parameter switches the entire storage layer.

    Parameters:
    -----------
    backend : str
        Storage backend type: "in_memory" | "redis"
    **kwargs
        Backend-specific configuration (e.g., redis host/port)

    Returns:
    --------
    storage : StorageManager
        Initialized storage backend instance

    Example:
    --------
    >>> # Prototype mode (default)
    >>> storage = create_storage("in_memory")
    >>>
    >>> # Production mode
    >>> storage = create_storage("redis", host="redis.example.com", port=6379)
    """
    if backend == "in_memory":
        return InMemoryStorage()
    elif backend == "redis":
        return RedisStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage backend: {backend}. Use 'in_memory' or 'redis'.")


# ============================================================================
# Configuration
# ============================================================================

# Default storage backend for the application
# Change this to "redis" for production deployment
DEFAULT_STORAGE_BACKEND = "in_memory"

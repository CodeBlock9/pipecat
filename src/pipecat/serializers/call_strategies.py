"""Strategy interfaces for call operations in serializers.

This module defines the abstract interfaces that telephony serializers can use
to delegate call operations (transfer, hangup) to provider-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

# Total bound, in seconds, on a carrier REST request a serializer makes itself
# to end a call. It runs inside the terminal frame's traversal, so without one
# (aiohttp's default total is 300 s) a carrier that stops answering holds the
# pipeline's end for as long.
CARRIER_REQUEST_TIMEOUT_SECS = 5.0


class CallOperationStrategy(ABC):
    """Base strategy for call operations."""

    pass


class TransferStrategy(CallOperationStrategy):
    """Strategy for handling call transfer operations.

    Implementations should handle all aspects of transferring a call.
    """

    @abstractmethod
    async def execute_transfer(self, context: dict[str, Any]) -> bool:
        """Execute call transfer with provider-specific logic.

        Args:
            context: Dictionary containing all necessary transfer context:
                - Provider-specific connection details
                - Call identifiers
                - Transfer destination information
                - Any other context needed for the operation

        Returns:
            bool: True if transfer was successful, False otherwise
        """
        pass


class HangupStrategy(CallOperationStrategy):
    """Strategy for handling call hangup operations."""

    @abstractmethod
    async def execute_hangup(self, context: dict[str, Any]) -> bool:
        """Execute call hangup with provider-specific logic.

        Args:
            context: Dictionary containing all necessary hangup context:
                - Provider-specific connection details
                - Call identifiers
                - Any other context needed for the operation

        Returns:
            bool: True if hangup was successful, False otherwise
        """
        pass

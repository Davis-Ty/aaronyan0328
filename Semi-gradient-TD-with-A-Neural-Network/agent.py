#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for RL agents.

    Defines the required methods for interacting with an RL-Glue environment.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def agent_init(self, agent_info=None):
        """Initialize the agent.

        Args:
            agent_info (dict, optional): Agent initialization information. Defaults to None.
        """
        pass

    @abstractmethod
    def agent_start(self, observation: np.ndarray) -> np.ndarray:
        """Start an episode.

        Args:
            observation (np.ndarray): Initial environment observation.

        Returns:
            np.ndarray: The agent's first action.
        """
        pass

    @abstractmethod
    def agent_step(self, reward: float, observation: np.ndarray) -> np.ndarray:
        """Take a step in the environment.

        Args:
            reward (float): Reward received from the previous step.
            observation (np.ndarray): Current environment observation.

        Returns:
            np.ndarray: The agent's next action.
        """
        pass

    @abstractmethod
    def agent_end(self, reward: float):
        """End an episode.

        Args:
            reward (float): Reward received for the final step.
        """
        pass

    @abstractmethod
    def agent_cleanup(self):
        """Clean up agent resources."""
        pass

    @abstractmethod
    def agent_message(self, message: str) -> str:
        """Handle messages from the experiment.

        Args:
            message (str): Message from the experiment.

        Returns:
            str: Response to the message.
        """
        pass

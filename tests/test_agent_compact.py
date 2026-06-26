"""Test Agent compact logic."""

import unittest
from unittest.mock import Mock, patch
from app.agent.agent import TranslationAgent
from app.translation.client import ClientConfig


class TestAgentCompact(unittest.TestCase):
    
    def setUp(self):
        """Create a mock agent."""
        config = ClientConfig(
            base_url="http://mock",
            api_key="mock_key",
            model="mock_model"
        )
        self.agent = TranslationAgent(config)
    
    def test_max_history_pairs_constant(self):
        """Test MAX_HISTORY_PAIRS constant exists."""
        self.assertEqual(self.agent.MAX_HISTORY_PAIRS, 10)
    
    def test_initial_messages_empty(self):
        """Test messages starts empty."""
        self.assertEqual(len(self.agent.messages), 0)
    
    def test_compact_preserves_system_and_confirmation(self):
        """Test compact keeps first 3 messages."""
        # Simulate digest_rules result
        self.agent.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Confirm"},
            {"role": "assistant", "content": "Confirmed"}
        ]
        
        # Add 15 translation pairs (30 messages)
        for i in range(15):
            self.agent.messages.append({"role": "user", "content": f"Text {i}"})
            self.agent.messages.append({"role": "assistant", "content": f"Translation {i}"})
        
        # Total: 3 + 30 = 33 messages
        self.assertEqual(len(self.agent.messages), 33)
        
        self.agent._compact_messages()
        
        # After compact: 3 + 20 = 23
        self.assertEqual(len(self.agent.messages), 23)
        
        # Check system preserved
        self.assertEqual(self.agent.messages[0]["role"], "system")
        
        # Check confirmation preserved
        self.assertEqual(self.agent.messages[1]["role"], "user")
        self.assertEqual(self.agent.messages[2]["role"], "assistant")
        
        # Check last message is from the most recent translations
        self.assertEqual(self.agent.messages[-1]["content"], "Translation 14")
    
    def test_conflict_handling_in_system_prompt(self):
        """Test that absolute constraints get a conflict handling clause."""
        with patch.object(self.agent._thinking_client, 'chat', return_value="Confirmed"):
            self.agent.digest_rules("Original prompt：输出只能使用平假名。")
            
            # Check system message contains conflict handling
            system_msg = self.agent.messages[0]["content"]
            self.assertIn("无法转换", system_msg)
            self.assertIn("英文、数字、符号", system_msg)
            self.assertIn("Original prompt", system_msg)


if __name__ == '__main__':
    unittest.main()

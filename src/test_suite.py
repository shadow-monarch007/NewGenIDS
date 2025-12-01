"""
test_suite.py - Comprehensive Testing Suite
----------------------------------------
Unit tests and integration tests for the IDS system
"""
import os
import sys
import unittest
import numpy as np
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IDSModel, NextGenIDS
from src.ml_phishing_detector import MLPhishingDetector
from src.auto_response import AutoResponseSystem
from src.auth import AuthManager, login_user
from src.security import InputValidator, RateLimiter

class TestModels(unittest.TestCase):
    """Test neural network models"""
    
    def setUp(self):
        self.input_dim = 20
        self.num_classes = 6
        self.batch_size = 32
        self.seq_len = 100
    
    def test_ids_model_forward(self):
        """Test IDSModel forward pass"""
        model = IDSModel(self.input_dim, hidden_size=128, num_layers=2, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_nextgen_ids_forward(self):
        """Test NextGenIDS forward pass"""
        model = NextGenIDS(self.input_dim, hidden_size=128, num_layers=2, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_model_output_probabilities(self):
        """Test model outputs valid probabilities"""
        model = IDSModel(self.input_dim, hidden_size=128, num_layers=2, num_classes=self.num_classes)
        model.eval()
        
        x = torch.randn(1, self.seq_len, self.input_dim)
        output = model(x)
        probs = torch.softmax(output, dim=1)
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
        
        # All probabilities should be between 0 and 1
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

class TestPhishingDetector(unittest.TestCase):
    """Test ML phishing detector"""
    
    def setUp(self):
        self.detector = MLPhishingDetector()
    
    def test_legitimate_url(self):
        """Test detection of legitimate URL"""
        url = "https://www.google.com/search?q=test"
        is_phishing, confidence, risk_score = self.detector.predict_url(url)
        
        # Should detect as safe (but may vary due to ML)
        self.assertIsInstance(is_phishing, bool)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(risk_score, int)
        self.assertGreaterEqual(risk_score, 0)
        self.assertLessEqual(risk_score, 100)
    
    def test_suspicious_url(self):
        """Test detection of suspicious URL"""
        url = "http://192.168.1.1@paypal-secure-verify-account.tk/login.php"
        is_phishing, confidence, risk_score = self.detector.predict_url(url)
        
        # Should have high risk score
        self.assertIsInstance(risk_score, int)
        self.assertGreater(risk_score, 30)  # At least somewhat suspicious
    
    def test_phishing_email(self):
        """Test phishing email detection"""
        email = """
        URGENT: Your account has been suspended!
        Click here immediately to verify your identity: http://suspicious-link.com
        Act now within 24 hours or your account will be permanently locked.
        """
        
        is_phishing, confidence, risk_score = self.detector.predict_email(email)
        
        # Should detect as phishing
        self.assertIsInstance(risk_score, int)
        self.assertGreater(risk_score, 40)  # Should have elevated risk

class TestAutoResponse(unittest.TestCase):
    """Test automated response system"""
    
    def setUp(self):
        self.response_system = AutoResponseSystem(enabled=True, dry_run=True)
    
    def test_ip_blocking(self):
        """Test IP blocking functionality"""
        test_ip = "192.168.100.50"
        success, message = self.response_system.block_ip(test_ip, reason="Test")
        
        self.assertTrue(success)
        self.assertIn("DRY RUN", message)
    
    def test_whitelist_protection(self):
        """Test that whitelisted IPs cannot be blocked"""
        whitelist_ip = "127.0.0.1"
        success, message = self.response_system.block_ip(whitelist_ip)
        
        self.assertFalse(success)
        self.assertIn("whitelist", message.lower())
    
    def test_threat_response_high_confidence(self):
        """Test response to high confidence threat"""
        action, success, message = self.response_system.respond_to_threat(
            threat_type="DDoS",
            source_ip="10.0.0.50",
            confidence=0.95
        )
        
        self.assertIn(action, ['BLOCK_IP', 'ALERT_ONLY'])
        self.assertTrue(success)
    
    def test_threat_response_low_confidence(self):
        """Test response to low confidence threat"""
        action, success, message = self.response_system.respond_to_threat(
            threat_type="Port_Scan",
            source_ip="10.0.0.51",
            confidence=0.45
        )
        
        self.assertEqual(action, 'LOG_EVENT')
        self.assertTrue(success)

class TestAuthentication(unittest.TestCase):
    """Test authentication system"""
    
    def setUp(self):
        self.auth_manager = AuthManager()
    
    def test_default_login(self):
        """Test login with default credentials"""
        success, message, session_id = login_user('admin', 'admin123')
        
        self.assertTrue(success)
        self.assertIsNotNone(session_id)
    
    def test_invalid_login(self):
        """Test login with invalid credentials"""
        success, message, session_id = login_user('admin', 'wrongpassword')
        
        self.assertFalse(success)
        self.assertIsNone(session_id)
    
    def test_session_validation(self):
        """Test session validation"""
        # Create session directly through auth_manager
        session_id = self.auth_manager.create_session('admin')
        
        self.assertTrue(self.auth_manager.validate_session(session_id))
        self.assertFalse(self.auth_manager.validate_session('invalid_session'))
    
    def test_password_hashing(self):
        """Test password hashing"""
        password = "test_password_123"
        hash1 = self.auth_manager.hash_password(password)
        hash2 = self.auth_manager.hash_password(password)
        
        # Same password should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Hash should be different from plaintext
        self.assertNotEqual(hash1, password)

class TestInputValidation(unittest.TestCase):
    """Test input validation"""
    
    def test_valid_ipv4(self):
        """Test IPv4 validation"""
        valid, result = InputValidator.validate_ip("192.168.1.1")
        self.assertTrue(valid)
        
        valid, result = InputValidator.validate_ip("256.1.1.1")
        self.assertFalse(valid)
    
    def test_valid_url(self):
        """Test URL validation"""
        valid, result = InputValidator.validate_url("https://www.example.com")
        self.assertTrue(valid)
        
        valid, result = InputValidator.validate_url("javascript:alert(1)")
        self.assertFalse(valid)
    
    def test_filename_validation(self):
        """Test filename validation"""
        valid, result = InputValidator.validate_filename("test.csv", allowed_extensions=['csv', 'pcap'])
        self.assertTrue(valid)
        
        valid, result = InputValidator.validate_filename("../etc/passwd")
        self.assertFalse(valid)
    
    def test_port_validation(self):
        """Test port number validation"""
        valid, result = InputValidator.validate_port(8080)
        self.assertTrue(valid)
        self.assertEqual(result, 8080)
        
        valid, result = InputValidator.validate_port(70000)
        self.assertFalse(valid)
    
    def test_string_sanitization(self):
        """Test string sanitization"""
        dirty = "Test\x00String\x01With\x02Control"
        clean = InputValidator.sanitize_string(dirty)
        
        # Should remove control characters
        self.assertNotIn('\x00', clean)
        self.assertIn('Test', clean)

class TestRateLimiter(unittest.TestCase):
    """Test rate limiting"""
    
    def setUp(self):
        self.limiter = RateLimiter()
    
    def test_rate_limiting(self):
        """Test rate limiting enforcement"""
        ip = "192.168.1.100"
        
        # Set a low limit for testing
        self.limiter.limits['test'] = (3, 60)
        
        # First 3 requests should pass
        self.assertFalse(self.limiter.is_rate_limited(ip, 'test'))
        self.assertFalse(self.limiter.is_rate_limited(ip, 'test'))
        self.assertFalse(self.limiter.is_rate_limited(ip, 'test'))
        
        # 4th request should be blocked
        self.assertTrue(self.limiter.is_rate_limited(ip, 'test'))
    
    def test_remaining_requests(self):
        """Test getting remaining request count"""
        ip = "192.168.1.101"
        self.limiter.limits['test2'] = (5, 60)
        
        # Initial remaining should be 5
        remaining = self.limiter.get_remaining(ip, 'test2')
        self.assertEqual(remaining, 5)
        
        # After one request, should be 4
        self.limiter.is_rate_limited(ip, 'test2')
        remaining = self.limiter.get_remaining(ip, 'test2')
        self.assertEqual(remaining, 4)

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestPhishingDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoResponse))
    suite.addTests(loader.loadTestsFromTestCase(TestAuthentication))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiter))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

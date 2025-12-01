"""
ml_phishing_detector.py - Machine Learning Phishing Detection
-----------------------------------------------------------
Uses Random Forest classifier trained on URL/email features
"""
import re
import pickle
import os
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class MLPhishingDetector:
    """ML-based phishing detector with feature engineering"""
    
    def __init__(self, model_path='models/phishing_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self._load_or_train_model()
    
    def _extract_url_features(self, url):
        """Extract numerical features from URL"""
        features = {}
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            
            # Length features
            features['url_length'] = len(url)
            features['domain_length'] = len(domain)
            features['path_length'] = len(path)
            
            # Count features
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_digits'] = sum(c.isdigit() for c in url)
            features['num_special_chars'] = sum(not c.isalnum() for c in url)
            
            # Suspicious patterns
            features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0
            features['has_at_symbol'] = 1 if '@' in url else 0
            features['has_double_slash'] = 1 if '//' in path else 0
            features['subdomain_count'] = domain.count('.') - 1 if domain else 0
            
            # TLD check
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work']
            features['suspicious_tld'] = 1 if any(url.endswith(tld) for tld in suspicious_tlds) else 0
            
            # HTTPS
            features['is_https'] = 1 if parsed.scheme == 'https' else 0
            
            # Brand keywords (potential spoofing)
            brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook', 'bank', 'secure', 'login', 'account', 'verify']
            features['has_brand_keyword'] = 1 if any(brand in url.lower() for brand in brands) else 0
            
        except:
            # If parsing fails, return safe defaults
            features = {k: 0 for k in ['url_length', 'domain_length', 'path_length', 'num_dots', 
                                       'num_hyphens', 'num_underscores', 'num_slashes', 'num_digits',
                                       'num_special_chars', 'has_ip', 'has_at_symbol', 'has_double_slash',
                                       'subdomain_count', 'suspicious_tld', 'is_https', 'has_brand_keyword']}
        
        return features
    
    def _create_training_data(self):
        """Create synthetic training data for demonstration"""
        # Legitimate URLs features
        legit_samples = [
            {'url_length': 35, 'domain_length': 15, 'path_length': 10, 'num_dots': 2, 'num_hyphens': 0,
             'num_underscores': 0, 'num_slashes': 2, 'num_digits': 0, 'num_special_chars': 5,
             'has_ip': 0, 'has_at_symbol': 0, 'has_double_slash': 0, 'subdomain_count': 1,
             'suspicious_tld': 0, 'is_https': 1, 'has_brand_keyword': 0},
        ] * 100  # Repeat for more samples
        
        # Phishing URLs features
        phish_samples = [
            {'url_length': 75, 'domain_length': 50, 'path_length': 20, 'num_dots': 5, 'num_hyphens': 3,
             'num_underscores': 2, 'num_slashes': 4, 'num_digits': 8, 'num_special_chars': 15,
             'has_ip': 1, 'has_at_symbol': 1, 'has_double_slash': 1, 'subdomain_count': 4,
             'suspicious_tld': 1, 'is_https': 0, 'has_brand_keyword': 1},
        ] * 100
        
        # Add variation
        X_legit = []
        for sample in legit_samples:
            varied = sample.copy()
            # Add random variation
            for key in varied:
                if key not in ['has_ip', 'has_at_symbol', 'has_double_slash', 'suspicious_tld', 'is_https', 'has_brand_keyword']:
                    varied[key] += np.random.randint(-5, 5)
                    varied[key] = max(0, varied[key])
            X_legit.append(list(varied.values()))
        
        X_phish = []
        for sample in phish_samples:
            varied = sample.copy()
            for key in varied:
                if key not in ['has_ip', 'has_at_symbol', 'has_double_slash', 'suspicious_tld', 'is_https', 'has_brand_keyword']:
                    varied[key] += np.random.randint(-5, 5)
                    varied[key] = max(0, varied[key])
            X_phish.append(list(varied.values()))
        
        X = np.array(X_legit + X_phish)
        y = np.array([0] * len(X_legit) + [1] * len(X_phish))  # 0=legit, 1=phishing
        
        return X, y
    
    def _load_or_train_model(self):
        """Load existing model or train new one"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"[OK] Loaded phishing detection model from {self.model_path}")
                return
            except:
                print(f"[WARN] Failed to load model, training new one...")
        
        # Train new model
        X, y = self._create_training_data()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✓ Trained and saved new phishing detection model to {self.model_path}")
    
    def predict_url(self, url):
        """
        Predict if URL is phishing
        Returns: (is_phishing: bool, confidence: float, risk_score: int)
        """
        try:
            # Extract features
            features = self._extract_url_features(url)
            X = np.array([list(features.values())])
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            is_phishing = bool(prediction == 1)
            confidence = probabilities[1] if is_phishing else probabilities[0]
            risk_score = int(probabilities[1] * 100)
            
            return is_phishing, confidence, risk_score
            
        except Exception as e:
            print(f"Error predicting URL: {e}")
            return False, 0.0, 0
    
    def predict_email(self, email_content, sender=None):
        """
        Predict if email is phishing
        Returns: (is_phishing: bool, confidence: float, risk_score: int)
        """
        try:
            risk_score = 0
            indicators = []
            
            # Check for suspicious keywords
            phishing_keywords = [
                'urgent', 'verify', 'suspended', 'locked', 'unusual activity',
                'confirm your identity', 'click here', 'limited time', 'act now',
                'winner', 'prize', 'claim', 'refund', 'tax', 'IRS', 'payment failed'
            ]
            
            email_lower = email_content.lower()
            for keyword in phishing_keywords:
                if keyword in email_lower:
                    risk_score += 10
                    indicators.append(f"Suspicious keyword: {keyword}")
            
            # Check for urgency
            if re.search(r'(urgent|immediately|within \d+ hours|asap)', email_lower):
                risk_score += 15
                indicators.append("Urgency tactics")
            
            # Check for links
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_content)
            if urls:
                for url in urls:
                    url_is_phish, url_conf, url_score = self.predict_url(url)
                    if url_is_phish:
                        risk_score += 20
                        indicators.append(f"Suspicious link: {url}")
            
            # Check sender mismatch
            if sender and '@' in sender:
                domain = sender.split('@')[1]
                # Check if display name doesn't match email domain
                if any(brand in sender.lower() for brand in ['paypal', 'amazon', 'bank']) and \
                   not any(brand in domain.lower() for brand in ['paypal', 'amazon', 'bank']):
                    risk_score += 25
                    indicators.append("Sender domain mismatch")
            
            # Check for attachments keywords
            if re.search(r'(attachment|attached|invoice\.pdf|receipt\.zip)', email_lower):
                risk_score += 10
                indicators.append("Suspicious attachment reference")
            
            risk_score = min(risk_score, 100)
            is_phishing = risk_score >= 50
            confidence = risk_score / 100.0
            
            return is_phishing, confidence, risk_score
            
        except Exception as e:
            print(f"Error predicting email: {e}")
            return False, 0.0, 0

# Global instance
ml_phishing_detector = MLPhishingDetector()

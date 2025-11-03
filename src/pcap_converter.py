"""
PCAP to Traffic Features Converter
Converts network capture files (PCAP) to the 20-feature CSV format required by the IDS model.
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from scapy.all import rdpcap, IP, TCP, UDP, DNS, Raw
    from scapy.layers.http import HTTPRequest
except ImportError:
    print("Error: scapy not installed. Run: pip install scapy")
    sys.exit(1)


class PCAPConverter:
    """Convert PCAP files to traffic feature CSV format."""
    
    def __init__(self):
        self.feature_names = [
            'packet_rate', 'packet_size', 'byte_rate', 'flow_duration',
            'total_packets', 'total_bytes', 'entropy', 'port_scan_score',
            'syn_flag_count', 'ack_flag_count', 'fin_flag_count', 'rst_flag_count',
            'psh_flag_count', 'urg_flag_count', 'unique_src_ports', 'unique_dst_ports',
            'payload_entropy', 'dns_query_count', 'http_request_count', 'ssl_handshake_count'
        ]
    
    def calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        if not data or len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = Counter(data)
        total = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def extract_flow_features(self, packets, window_size=5.0):
        """
        Extract traffic features from packets grouped by time windows.
        
        Args:
            packets: List of scapy packets
            window_size: Time window in seconds for grouping packets
            
        Returns:
            DataFrame with 20 traffic features
        """
        if not packets:
            return pd.DataFrame(columns=self.feature_names)
        
        print(f"  Processing {len(packets)} packets...")
        
        # Group packets by time windows
        flows = defaultdict(list)
        start_time = packets[0].time
        
        for pkt in packets:
            window_idx = int((pkt.time - start_time) / window_size)
            flows[window_idx].append(pkt)
        
        print(f"  Created {len(flows)} time windows ({window_size}s each)")
        
        # Extract features for each flow
        features_list = []
        
        for window_idx, window_packets in flows.items():
            features = self._extract_window_features(window_packets, window_size)
            features_list.append(features)
        
        df = pd.DataFrame(features_list, columns=self.feature_names)
        print(f"  Extracted features: {df.shape[0]} flows with {df.shape[1]} features")
        
        return df
    
    def _extract_window_features(self, packets, window_size):
        """Extract 20 features from a window of packets."""
        
        # Initialize counters
        total_packets = len(packets)
        total_bytes = 0
        syn_count = ack_count = fin_count = rst_count = psh_count = urg_count = 0
        src_ports = set()
        dst_ports = set()
        dns_queries = 0
        http_requests = 0
        ssl_handshakes = 0
        payload_data = b''
        packet_sizes = []
        
        # Process each packet
        for pkt in packets:
            # Packet size
            pkt_len = len(pkt)
            packet_sizes.append(pkt_len)
            total_bytes += pkt_len
            
            # TCP flags
            if TCP in pkt:
                flags = pkt[TCP].flags
                if flags & 0x02:  # SYN
                    syn_count += 1
                if flags & 0x10:  # ACK
                    ack_count += 1
                if flags & 0x01:  # FIN
                    fin_count += 1
                if flags & 0x04:  # RST
                    rst_count += 1
                if flags & 0x08:  # PSH
                    psh_count += 1
                if flags & 0x20:  # URG
                    urg_count += 1
                
                # Ports
                src_ports.add(pkt[TCP].sport)
                dst_ports.add(pkt[TCP].dport)
                
                # SSL detection (port 443)
                if pkt[TCP].dport == 443 or pkt[TCP].sport == 443:
                    ssl_handshakes += 1
            
            # UDP ports
            if UDP in pkt:
                src_ports.add(pkt[UDP].sport)
                dst_ports.add(pkt[UDP].dport)
            
            # DNS detection
            if DNS in pkt:
                dns_queries += 1
            
            # HTTP detection
            if HTTPRequest in pkt or (TCP in pkt and (pkt[TCP].dport == 80 or pkt[TCP].sport == 80)):
                http_requests += 1
            
            # Payload for entropy calculation
            if Raw in pkt:
                payload_data += bytes(pkt[Raw].load[:100])  # Limit to 100 bytes per packet
        
        # Calculate features
        avg_packet_size = np.mean(packet_sizes) if packet_sizes else 0
        packet_rate = total_packets / window_size if window_size > 0 else 0
        byte_rate = total_bytes / window_size if window_size > 0 else 0
        
        # Entropy calculations
        overall_entropy = self.calculate_entropy(payload_data) if payload_data else 0
        payload_entropy = overall_entropy
        
        # Port scan detection (many unique dst ports = potential scan)
        port_scan_score = len(dst_ports) / max(total_packets, 1)
        
        # Build feature vector
        features = [
            packet_rate,           # 0: packet_rate
            avg_packet_size,       # 1: packet_size
            byte_rate,             # 2: byte_rate
            window_size,           # 3: flow_duration
            total_packets,         # 4: total_packets
            total_bytes,           # 5: total_bytes
            overall_entropy,       # 6: entropy
            port_scan_score,       # 7: port_scan_score
            syn_count,             # 8: syn_flag_count
            ack_count,             # 9: ack_flag_count
            fin_count,             # 10: fin_flag_count
            rst_count,             # 11: rst_flag_count
            psh_count,             # 12: psh_flag_count
            urg_count,             # 13: urg_flag_count
            len(src_ports),        # 14: unique_src_ports
            len(dst_ports),        # 15: unique_dst_ports
            payload_entropy,       # 16: payload_entropy
            dns_queries,           # 17: dns_query_count
            http_requests,         # 18: http_request_count
            ssl_handshakes         # 19: ssl_handshake_count
        ]
        
        return features
    
    def convert_pcap_to_csv(self, pcap_file, output_csv=None, window_size=5.0, max_packets=None):
        """
        Convert PCAP file to traffic features CSV.
        
        Args:
            pcap_file: Path to PCAP file
            output_csv: Path to output CSV (default: same name as PCAP with .csv)
            window_size: Time window in seconds for grouping packets
            max_packets: Maximum number of packets to process (None = all)
            
        Returns:
            Path to created CSV file
        """
        if not os.path.exists(pcap_file):
            raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
        
        # Default output filename
        if output_csv is None:
            base_name = os.path.splitext(pcap_file)[0]
            output_csv = f"{base_name}_features.csv"
        
        print(f"\n{'='*70}")
        print(f"🔄 Converting PCAP to Traffic Features")
        print(f"{'='*70}")
        print(f"📂 Input:  {pcap_file}")
        print(f"📊 Output: {output_csv}")
        print(f"⏱️  Window: {window_size}s")
        
        # Read PCAP file
        print(f"\n📖 Reading PCAP file...")
        try:
            packets = rdpcap(pcap_file)
            if max_packets:
                packets = packets[:max_packets]
                print(f"  Limiting to first {max_packets} packets")
        except Exception as e:
            print(f"❌ Error reading PCAP: {e}")
            raise
        
        print(f"✅ Loaded {len(packets)} packets")
        
        # Extract features
        print(f"\n🔍 Extracting traffic features...")
        df = self.extract_flow_features(packets, window_size)
        
        if df.empty:
            print(f"⚠️  Warning: No features extracted!")
            return None
        
        # Save to CSV
        print(f"\n💾 Saving to CSV...")
        df.to_csv(output_csv, index=False)
        print(f"✅ Created: {output_csv}")
        print(f"   Shape: {df.shape[0]} flows × {df.shape[1]} features")
        
        # Show statistics
        print(f"\n📊 Traffic Statistics:")
        print(f"   Total flows: {len(df)}")
        print(f"   Avg packet rate: {df['packet_rate'].mean():.2f} packets/s")
        print(f"   Avg packet size: {df['packet_size'].mean():.2f} bytes")
        print(f"   Total packets: {df['total_packets'].sum():.0f}")
        print(f"   Total bytes: {df['total_bytes'].sum():.0f}")
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS! PCAP converted to traffic features CSV")
        print(f"{'='*70}")
        print(f"\n💡 You can now upload '{output_csv}' to the dashboard!")
        
        return output_csv


def main():
    """Command-line interface for PCAP converter."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert PCAP files to traffic feature CSV for IDS analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pcap_converter.py capture.pcap
  python src/pcap_converter.py capture.pcap --output traffic.csv
  python src/pcap_converter.py capture.pcap --window 10 --max-packets 10000
        """
    )
    
    parser.add_argument('pcap_file', help='Path to PCAP file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--window', '-w', type=float, default=5.0,
                       help='Time window size in seconds (default: 5.0)')
    parser.add_argument('--max-packets', '-m', type=int,
                       help='Maximum packets to process (default: all)')
    
    args = parser.parse_args()
    
    # Convert PCAP
    converter = PCAPConverter()
    try:
        output_file = converter.convert_pcap_to_csv(
            args.pcap_file,
            args.output,
            args.window,
            args.max_packets
        )
        
        if output_file:
            print(f"\n🎯 Next step: Upload {output_file} to the dashboard at http://localhost:5000")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Download Sample PCAP Files for Testing
Downloads real network capture files from public repositories.
"""

import os
import requests

def download_file(url, output_path):
    """Download file from URL."""
    print(f"📥 Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"✅ Downloaded: {output_path} ({file_size:.2f} KB)")
    return output_path


def download_sample_pcaps():
    """Download sample PCAP files for testing."""
    
    # Create directory
    pcap_dir = "downloads/sample_pcaps"
    os.makedirs(pcap_dir, exist_ok=True)
    
    print("="*70)
    print("🌐 Downloading Sample PCAP Files for Testing")
    print("="*70)
    
    # Sample PCAP files from Wireshark wiki and other sources
    samples = [
        {
            "name": "http_sample.pcap",
            "url": "https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/http.cap",
            "description": "HTTP traffic capture"
        },
        {
            "name": "dns_sample.pcap",
            "url": "https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/dns.cap",
            "description": "DNS queries capture"
        },
        {
            "name": "telnet_sample.pcap",
            "url": "https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/telnet-raw.pcap",
            "description": "Telnet session capture"
        }
    ]
    
    downloaded = []
    
    for sample in samples:
        try:
            output_path = os.path.join(pcap_dir, sample["name"])
            print(f"\n📦 {sample['description']}")
            download_file(sample["url"], output_path)
            downloaded.append(output_path)
        except Exception as e:
            print(f"❌ Failed to download {sample['name']}: {e}")
    
    print("\n" + "="*70)
    print(f"✅ Downloaded {len(downloaded)} PCAP files")
    print("="*70)
    
    if downloaded:
        print("\n📁 Files saved in: downloads/sample_pcaps/")
        for file in downloaded:
            print(f"   - {os.path.basename(file)}")
        
        print("\n🔄 Next Steps:")
        print("   1. Convert PCAP to CSV:")
        print("      python src/pcap_converter.py downloads/sample_pcaps/http_sample.pcap")
        print("\n   2. Or upload directly to dashboard (auto-converts):")
        print("      http://localhost:5000")
    
    return downloaded


if __name__ == '__main__':
    try:
        files = download_sample_pcaps()
        if files:
            print(f"\n🎯 Ready to test! Upload PCAP files to the dashboard.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

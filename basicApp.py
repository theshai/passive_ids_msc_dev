from scapy.all import get_if_list
from scapy.all import sniff

def packet_handler(pkt):
    print(pkt.summary())


print("listening for packets...")

sniff(iface="eth0",prn=packet_handler,store=False)
#sniff(iface="wlan0",prn=packet_handler,store=False)


"""AmbuSmart - authenticated GPS telemetry packet (vehicle -> intersection).

Layout (big-endian, 61 bytes):
  vehicle_id u32 | timestamp_ms u64 | lat f64 | lon f64 | heading_ddeg u16 | speed_dkmh u16 | flags u8
  | hmac_sha256 (32 bytes over the 29-byte payload)
"""
import hashlib
import hmac
import struct

FMT = ">IQddHHB"
PAYLOAD_LEN = struct.calcsize(FMT)


def pack(key: bytes, vehicle_id, ts_ms, lat, lon, heading_deg, speed_kmh, emergency=True) -> bytes:
    payload = struct.pack(FMT, vehicle_id, ts_ms, lat, lon, int(round(heading_deg * 10)) % 3600,
                          int(round(speed_kmh * 10)), 1 if emergency else 0)
    return payload + hmac.new(key, payload, hashlib.sha256).digest()


def verify_and_unpack(key: bytes, pkt: bytes):
    """Return a dict for a valid packet, or None if length or HMAC check fails."""
    if len(pkt) != PAYLOAD_LEN + 32:
        return None
    payload, tag = pkt[:PAYLOAD_LEN], pkt[PAYLOAD_LEN:]
    if not hmac.compare_digest(hmac.new(key, payload, hashlib.sha256).digest(), tag):
        return None
    vid, ts, lat, lon, hd, sp, fl = struct.unpack(FMT, payload)
    return dict(vehicle_id=vid, ts_ms=ts, lat=lat, lon=lon, heading_deg=hd / 10, speed_kmh=sp / 10,
                emergency=bool(fl & 1))

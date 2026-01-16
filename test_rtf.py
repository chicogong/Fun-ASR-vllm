#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Fun-ASR vLLM API RTF (Real Time Factor)
Supports single request and batch request statistics
"""
import argparse
import base64
import json
import time
from pathlib import Path
from typing import List, Optional

import requests
import torchaudio


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def test_single_request(
    api_url: str,
    audio_path: str,
    language: Optional[str] = None,
) -> dict:
    """Single request test"""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    payload = {
        "audio_b64": audio_b64,
        "audio_format": Path(audio_path).suffix.lstrip("."),
        "language": language,
        "itn": True,
        "hotwords": [],
    }
    
    start = time.perf_counter()
    resp = requests.post(f"{api_url}/asr", json=payload, timeout=300)
    client_elapsed = time.perf_counter() - start
    
    if resp.status_code != 200:
        return {"error": resp.text, "status_code": resp.status_code}
    
    result = resp.json()
    result["client_latency_sec"] = client_elapsed
    return result


def test_batch_requests(
    api_url: str,
    audio_paths: List[str],
    language: Optional[str] = None,
    num_requests: int = 10,
) -> dict:
    """Batch request test, calculate average RTF"""
    results = []
    total_audio_duration = 0.0
    total_latency = 0.0
    
    for i in range(num_requests):
        audio_path = audio_paths[i % len(audio_paths)]
        print(f"[{i+1}/{num_requests}] Testing with {audio_path}...")
        
        result = test_single_request(api_url, audio_path, language)
        if "error" in result:
            print(f"  [FAIL] Error: {result['error']}")
            continue
        
        audio_dur = result.get("audio_duration_sec", 0) or get_audio_duration(audio_path)
        latency = result.get("latency_sec", result.get("client_latency_sec", 0))
        rtf = result.get("rtf")
        
        total_audio_duration += audio_dur
        total_latency += latency
        
        print(f"  [OK] text: {result.get('text', '')[:50]}...")
        print(f"     audio: {audio_dur:.2f}s, latency: {latency:.3f}s, RTF: {rtf:.4f}" if rtf else f"     audio: {audio_dur:.2f}s, latency: {latency:.3f}s")
        
        results.append({
            "audio_path": audio_path,
            "audio_duration_sec": audio_dur,
            "latency_sec": latency,
            "rtf": rtf,
            "text": result.get("text", ""),
        })
    
    # Summary statistics
    avg_rtf = total_latency / total_audio_duration if total_audio_duration > 0 else None
    hours_per_day = (24 * 3600) / avg_rtf / 3600 if avg_rtf else None
    
    summary = {
        "num_requests": len(results),
        "total_audio_duration_sec": total_audio_duration,
        "total_latency_sec": total_latency,
        "avg_rtf": avg_rtf,
        "hours_per_day": hours_per_day,
        "results": results,
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test Fun-ASR vLLM API RTF")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080", help="API base URL")
    parser.add_argument("--audio", type=str, nargs="+", required=True, help="Audio file path(s)")
    parser.add_argument("--language", type=str, default=None, help="Language (e.g., Chinese, English)")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests for batch test")
    parser.add_argument("--single", action="store_true", help="Run single request test only")
    args = parser.parse_args()
    
    # Validate audio files
    audio_paths = []
    for p in args.audio:
        if Path(p).exists():
            audio_paths.append(p)
        else:
            print(f"[WARN] Audio file not found: {p}")
    
    if not audio_paths:
        print("[ERROR] No valid audio files found!")
        return
    
    print(f"[INFO] API URL: {args.api_url}")
    print(f"[INFO] Audio files: {audio_paths}")
    print(f"[INFO] Language: {args.language or 'auto'}")
    print()
    
    if args.single:
        # 单次请求
        result = test_single_request(args.api_url, audio_paths[0], args.language)
        print("=" * 60)
        print("Single Request Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 批量请求
        summary = test_batch_requests(
            args.api_url, audio_paths, args.language, args.num_requests
        )
        
        print()
        print("=" * 60)
        print("RTF Test Summary")
        print("=" * 60)
        print(f"  Total requests:        {summary['num_requests']}")
        print(f"  Total audio duration:  {summary['total_audio_duration_sec']:.2f} sec")
        print(f"  Total processing time: {summary['total_latency_sec']:.2f} sec")
        print(f"  Average RTF:           {summary['avg_rtf']:.4f}" if summary['avg_rtf'] else "  Average RTF:           N/A")
        print()
        if summary['hours_per_day']:
            print(f"  Daily processing capacity (24h): {summary['hours_per_day']:.1f} hours of audio")
            print(f"     (~{summary['hours_per_day'] * 60:.0f} minutes / ~{summary['hours_per_day'] * 3600:.0f} seconds)")
        print("=" * 60)


if __name__ == "__main__":
    main()

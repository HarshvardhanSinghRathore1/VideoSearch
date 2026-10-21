"""
Sample Asset Generator Script
Synthesizes a clean educational test video asset 'data/videos/sample.mp4' on
Database Normalization and Data Mining with speech audio and timestamp alignment.
"""

import json
import os
import sys
import subprocess
import wave
import struct
import math
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO_SECTIONS = [
    {
        "start": 0.0,
        "end": 45.0,
        "title": "Introduction to Database Normalization",
        "text": "Welcome to this lecture on database design. Database normalization is the systematic process of organizing structured data in a relational database. The primary goal is to minimize data redundancy and eliminate anomalies such as insertion, update, and deletion anomalies. By decomposing large tables into smaller, well-structured relations, we maintain high data integrity."
    },
    {
        "start": 45.0,
        "end": 85.0,
        "title": "First Normal Form (1NF)",
        "text": "Let us examine First Normal Form, or 1NF. A relation satisfies First Normal Form if and only if all column values are atomic. This means every cell contains a single discrete value rather than sets, arrays, or repeating groups. Every row must also be uniquely identifiable through a primary key."
    },
    {
        "start": 85.0,
        "end": 130.0,
        "title": "Second Normal Form (2NF)",
        "text": "Moving to Second Normal Form, or 2NF. To achieve 2NF, a table must first be in 1NF, and secondly, all non-prime attributes must be fully functionally dependent on the primary key. This eliminates partial dependencies where an attribute depends on only a portion of a composite primary key."
    },
    {
        "start": 130.0,
        "end": 175.0,
        "title": "Third Normal Form (3NF)",
        "text": "Next is Third Normal Form, or 3NF. A table is in 3NF if it satisfies 2NF and has no transitive functional dependencies. A transitive dependency occurs when non-key attribute A determines non-key attribute B, which in turn determines non-key attribute C. 3NF resolves this by decomposing the attributes into separate tables."
    },
    {
        "start": 175.0,
        "end": 215.0,
        "title": "Boyce-Codd Normal Form (BCNF)",
        "text": "Boyce-Codd Normal Form, or BCNF, is a stricter extension of 3NF. In BCNF, for every non-trivial functional dependency X determines Y, X must strictly be a candidate superkey of the relation. This removes anomalies that standard 3NF may still permit when multiple candidate keys overlap."
    },
    {
        "start": 215.0,
        "end": 260.0,
        "title": "Data Preprocessing in Data Mining",
        "text": "Now turning to data mining principles. Before applying analytical models, data preprocessing is essential. Major preprocessing stages include data cleaning to handle missing values and noisy data, data integration from multiple sources, data transformation including numerical normalization and scaling, and data reduction."
    },
    {
        "start": 260.0,
        "end": 305.0,
        "title": "Clustering with K-Means",
        "text": "In unsupervised data mining, clustering groups unlabeled observations into homogeneous subsets. The K-Means algorithm partitions data into K predefined clusters by iteratively minimizing the sum of squared Euclidean distances between points and their corresponding cluster centroids."
    },
    {
        "start": 305.0,
        "end": 345.0,
        "title": "Normalization Trade-offs & Query Performance",
        "text": "Finally, we consider engineering trade-offs. While high normalization prevents insertion and update anomalies, it often requires frequent multi-table JOIN operations. In high-throughput analytical workloads, controlled denormalization is often applied to boost read query throughput."
    }
]


def generate_speech_audio(wav_path: Path):
    """Uses PowerShell System.Speech.Synthesis or generates synthetic acoustic speech wav."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    full_text = " ".join([s["text"] for s in VIDEO_SECTIONS])

    # Try PowerShell SpeechSynthesizer for realistic speech WAV
    ps_script = f"""
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.Rate = 0
    $synth.SetOutputToWaveFile('{str(wav_path.resolve())}')
    $synth.Speak('{full_text.replace("'", "''")}')
    $synth.Dispose()
    """
    try:
        res = subprocess.run(["powershell", "-Command", ps_script], capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 1000:
            print(f"Generated realistic SAPI speech audio at: {wav_path}")
            return
    except Exception as e:
        print(f"PowerShell TTS note: {e}")

    # Fallback tone-modulated speech WAV
    sample_rate = 16000
    total_duration = 345.0
    num_samples = int(sample_rate * total_duration)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # 220Hz carrier with harmonic speech modulation
            val = int(3000 * math.sin(2 * math.pi * 220 * t) * math.sin(2 * math.pi * 3 * t))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    print(f"Generated fallback audio at: {wav_path}")


def create_mp4_video(audio_path: Path, video_path: Path):
    """Muxes audio and video slide frames into MP4."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

    # Generate test video container with ffmpeg lavfi testsrc & audio
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "lavfi", "-i", "color=c=0x1E293B:s=1280x720:r=24:d=345",
        "-i", str(audio_path.resolve()),
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(video_path.resolve())
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"Generated sample video at: {video_path}")
    except Exception as e:
        print(f"FFmpeg MP4 creation note: {e}")
        # If libx264 is unavailable, copy audio to mock container or test video
        fallback_cmd = [
            ffmpeg_exe, "-y",
            "-f", "lavfi", "-i", "nullsrc=s=640x480:d=345",
            "-i", str(audio_path.resolve()),
            "-c:a", "copy",
            str(video_path.resolve())
        ]
        subprocess.run(fallback_cmd, capture_output=True)


def export_ground_truth_transcripts(transcripts_dir: Path):
    """Exports structured JSON and TXT transcripts for the sample video."""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON
    json_path = transcripts_dir / "sample_transcript.json"
    segments_data = [
        {
            "start": sec["start"],
            "end": sec["end"],
            "text": sec["text"]
        }
        for sec in VIDEO_SECTIONS
    ]
    data = {
        "source_video": "sample.mp4",
        "segment_count": len(VIDEO_SECTIONS),
        "total_duration_sec": 345.0,
        "metadata": {
            "title": "Database Normalization and Data Mining Principles",
            "speaker": "AI Lecturer",
            "sampling_rate_hz": 16000
        },
        "segments": segments_data
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 2. Formatted TXT
    txt_path = transcripts_dir / "sample_transcript.txt"
    lines = [
        "# Video Transcript: sample.mp4",
        "# Title: Database Normalization and Data Mining Principles",
        "# Segments: 8 | Total Duration: 00:05:45.00",
        "=" * 70,
        ""
    ]
    for sec in VIDEO_SECTIONS:
        from src.transcription import format_timestamp
        st = format_timestamp(sec["start"])
        et = format_timestamp(sec["end"])
        lines.append(f"[{st} - {et}] {sec['title']}")
        lines.append(f"{sec['text']}\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Exported sample transcripts to {json_path} and {txt_path}")


if __name__ == "__main__":
    v_dir = Path("data/videos")
    a_dir = Path("data/audio")
    t_dir = Path("data/transcripts")
    
    audio_file = a_dir / "sample.wav"
    video_file = v_dir / "sample.mp4"
    
    generate_speech_audio(audio_file)
    export_ground_truth_transcripts(t_dir)
    create_mp4_video(audio_file, video_file)

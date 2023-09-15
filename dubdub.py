from control_data import MX20, RAW_SAMPLES_DAY_1, RAW_SAMPLES_DAY_2, RAW_COMPRESSOR_DAY_1, RAW_COMPRESSOR_DAY_2
import torchaudio
import os
import subprocess

SR = 44100

def build_audio_file(mx20):
    # CXF = AudioSegment.silent(duration=0)
    # CYF = AudioSegment.silent(duration=0)
    D = os.listdir('.')
    for x in D:
        if 'tmp-split' in x and '.wav' in x:
            os.remove(x)
    XF = f'o_x_{mx20}.wav'
    YF = f'o_y_{mx20}.wav'
    N = 0
    for sarr, carr, day in [(RAW_SAMPLES_DAY_1, RAW_COMPRESSOR_DAY_1,'day1'), (RAW_SAMPLES_DAY_2,RAW_COMPRESSOR_DAY_2,'day2')]:
        for x in range(len(sarr)):
            print("ON", day, sarr[x], carr[x])
            #start = int(sarr[x] * 1000 / SR)
            #end = int(sarr[x+1] * 1000 / SR) if x != len(sarr) - 1 else -1
            # start = (sarr[x] * 1000 / SR)
            # end = (sarr[x+1] * 1000 / SR) if x != len(sarr) - 1 else -1
            start = sarr[x]
            end = sarr[x+1] if x != len(sarr) - 1 else None
            X, _ = torchaudio.load(f'/workspace/{day}/67_near.wav', frame_offset=start,num_frames=end-start if end != None else -1)
            Y1, _ = torchaudio.load(f'/workspace/{day}/67_MX20_1.wav', frame_offset=start,num_frames=end-start if end != None else -1)
            Y2, _ = torchaudio.load(f'/workspace/{day}/67_MX20_2.wav', frame_offset=start,num_frames=end-start if end != None else -1)
            if carr[x][1][0] == mx20:
                print('saving compressor 1')
                torchaudio.save(f'{N}-x-tmp-split.wav',X, sample_rate=44100)
                torchaudio.save(f'{N}-y-tmp-split.wav',Y1, sample_rate=44100)
                N += 1
            if carr[x][1][1] == mx20:
                print('saving compressor 2')
                torchaudio.save(f'{N}-x-tmp-split.wav',X, sample_rate=44100)
                torchaudio.save(f'{N}-y-tmp-split.wav',Y2, sample_rate=44100)
                N += 1
    XC = ' '.join([f'{x}-x-tmp-split.wav' for x in range(N)])
    YC = ' '.join([f'{x}-y-tmp-split.wav' for x in range(N)])
    subprocess.call(f'sox {XC} {XF}', shell=True)
    subprocess.call(f'sox {YC} {YF}', shell=True)

if __name__ == '__main__':
    build_audio_file(MX20.TWO)
    build_audio_file(MX20.FOUR)
    build_audio_file(MX20.EIGHT)
    build_audio_file(MX20.TWELVE)
// Subharmonic Synthesizer for Rockbox, by Vanessa, 2025

// This filter uses a silplistic sample and hold toggle method
// for creating signals with half the input frequency by effectively
// halving the samplerate and then upsampling again. The band to be
// processed is limited by a crossover low pass filter and output
// noise from aliasing is removed with another low pass filter.
// This filter tends to clip at higher gain because it adds entirely
// new signal components, so reducing the pregain is recommended.
//
// This code is licensed under the GNU General Public License version 2 (GPLv2).
//
// This software is provided "AS IS", without warranty of any kind,
// express or implied, including but not limited to the warranties
// of merchantability, fitness for a particular purpose and
// noninfringement. In no event shall the authors be liable for any
// claim, damages or other liability, whether in an action of contract,
// tort or otherwise, arising from, out of or in connection with the
// software or the use or other dealings in the software.

#include "rbcodecconfig.h"
#include "fixedpoint.h"
#include "dsp_proc_entry.h"
#include "dsp_core.h"
#include "settings.h"
#include "subharmonic.h"

// Memory variables for both low pass filters and the sample and hold toggle
static int32_t prev_crossover_out[2], subharmonic_hold_q16[2], prev_antialias_out_q16[2];
// Toggle state
static int toggle_state[2];
// default samplerate in case it cant be loaded from rockbox
static int samplerate = 44100;
// Filter coefficient and subharmonic gain
static int32_t alpha_q16 = 0, gain_q16 = 0;
// Pregain enable
static bool pregain;

// -24 .. +12 dB in 1-dB steps pre calculated as Q16 values
#define GAIN_TABLE_MIN_DB   -24
#define GAIN_TABLE_MAX_DB   12
static const int32_t gain_table_q16[GAIN_TABLE_MAX_DB - GAIN_TABLE_MIN_DB + 1] = {
    4145,    /* -24 dB */
    4655,    /* -23 dB */
    5226,    /* -22 dB */
    5867,    /* -21 dB */
    6588,    /* -20 dB */
    7399,    /* -19 dB */
    8310,    /* -18 dB */
    9336,    /* -17 dB */
    10488,   /* -16 dB */
    11782,   /* -15 dB */
    13234,   /* -14 dB */
    14865,   /* -13 dB */
    16700,   /* -12 dB */
    18766,   /* -11 dB */
    21095,   /* -10 dB */
    23721,   /* -9 dB  */
    26686,   /* -8 dB  */
    30033,   /* -7 dB  */
    33808,   /* -6 dB  */
    38065,   /* -5 dB  */
    42862,   /* -4 dB  */
    48265,   /* -3 dB  */
    54342,   /* -2 dB  */
    61172,   /* -1 dB  */
    65536,   /*  0 dB  */
    73690,   /* +1 dB  */
    82708,   /* +2 dB  */
    92713,   /* +3 dB  */
    103957,  /* +4 dB  */
    116607,  /* +5 dB  */
    130858,  /* +6 dB  */
    146928,  /* +7 dB  */
    165060,  /* +8 dB  */
    185533,  /* +9 dB  */
    208661,  /* +10 dB */
    234804,  /* +11 dB */
    264367   /* +12 dB */
};

// Reset filter state
static void flush_state(void)
{
    prev_crossover_out[0] = prev_crossover_out[1] = 0;
    subharmonic_hold_q16[0]  = subharmonic_hold_q16[1]  = 0;
    prev_antialias_out_q16[0] = prev_antialias_out_q16[1] = 0;
    toggle_state[0] = toggle_state[1] = 0;
}

// Recalculate all filter parameters
static void recompute(void)
{
    // Crossover
    // alpha = (2 pi f) / (2 pi f + fs)
    // f = fc, fs = samplerate

    const int32_t TWO_PI_Q16 = (int32_t)(6.283185307179586 * (1<<16)); // 2 pi in Q16 format
    int fc = global_settings.subharmonic_crossover;
    int64_t w_q16 = (int64_t)fc * TWO_PI_Q16; // calculate angular frequency in Q16 format
    int64_t den = w_q16 + ((int64_t)samplerate << 16);
    if (den <= 0) den = 1;
    alpha_q16 = (int32_t)((w_q16<<16) / den); // Low pass filter coefficient in Q16 format

    // Gain
    int db = global_settings.subharmonic_level;
    if (db < GAIN_TABLE_MIN_DB) db = GAIN_TABLE_MIN_DB;
    if (db > GAIN_TABLE_MAX_DB) db = GAIN_TABLE_MAX_DB;
    gain_q16 = gain_table_q16[db - GAIN_TABLE_MIN_DB];

    // Pregain
    pregain = global_settings.subharmonic_pregain;
}


// Register and enable or disable the subharmonic DSP process
static void dsp_set_subharmonic(void)
{
    struct dsp_config *dsp = dsp_get_config(CODEC_IDX_AUDIO);
    bool on = global_settings.subharmonic_enable;
    bool was_enabled = dsp_proc_enabled(dsp, DSP_PROC_SUBHARMONIC);

    if (on) // filter enabled
    {
        if (!was_enabled)   //  filter was not enabled yet
            dsp_proc_enable(dsp, DSP_PROC_SUBHARMONIC, true);
        else    // filter was already enabled
            dsp_configure(dsp, DSP_PROC_INIT, 1); // initialize filter again just in case

        // activate filter so it gets added to the DSP pipeline
        dsp_proc_activate(dsp, DSP_PROC_SUBHARMONIC, true);
        dsp_proc_set_in_place(dsp, DSP_PROC_SUBHARMONIC, true);
    }
    else    // filter diasabled
    {
        // deactivate filter
        dsp_proc_activate(dsp, DSP_PROC_SUBHARMONIC, false);
        if (was_enabled)    // disable filter if it was previously enabled
            dsp_proc_enable(dsp, DSP_PROC_SUBHARMONIC, false);
    }
}


// Main filter process
static void process(struct dsp_proc_entry *this, struct dsp_buffer **buf_p)
{
    (void)this;
    if (!global_settings.subharmonic_enable)    // was this accidentally called?
        return; // skip this filter if disabled

    struct dsp_buffer *buf = *buf_p;
    int32_t *sl = buf->p32[0];  // channel 0 -> left side
    int32_t *sr = buf->p32[1];  // channel 1 -> right side
    int count = buf->remcount;  // number of samples in buffer

    // process each sample
    for (int i = 0; i < count; i++)
    {
        // process each channel per sample (max 2 channels supported)
        for (int ch = 0; ch < buf->format.num_channels && ch < 2; ch++)
        {
            // get the individual sample for current loop run
            int32_t *samp = (ch==0) ? &sl[i] : &sr[i];
            int32_t sample = *samp;

            // Crossover low pass filter stage
            // y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
            int32_t crossover_out = (int32_t)((((int64_t)alpha_q16 * sample) +
                               (((int64_t)((1<<16)-alpha_q16)) * prev_crossover_out[ch])) >> 16);
            prev_crossover_out[ch] = crossover_out;
            
            // Subharmonic generator stage
            // only update stage output at every second sample
            if (!toggle_state[ch]) subharmonic_hold_q16[ch] = crossover_out;
            toggle_state[ch] ^= 1;  // update filter state
            int32_t subharmonic_out = subharmonic_hold_q16[ch];

            // Aliasing noise suppression low pass filter stage
            // y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
            int32_t antialias_out = (int32_t)((((int64_t)alpha_q16 * subharmonic_out) +
                               (((int64_t)((1<<16)-alpha_q16)) * prev_antialias_out_q16[ch])) >> 16);
            prev_antialias_out_q16[ch] = antialias_out;

            // Mixing output signal from subharmonic component and original signal
            int64_t mixer_out = 0;
            if (pregain) {  // -6 dB pregain enabled
                mixer_out = ((int64_t)sample>>1) + (((int64_t)gain_q16 * antialias_out)>>16);
            }
            else {  // no pregain
                mixer_out = (int64_t)sample + (((int64_t)gain_q16 * antialias_out)>>16);
            }

            // Clipping samples down to 32 bit
            if (mixer_out > INT32_MAX) mixer_out = INT32_MAX;
            if (mixer_out < INT32_MIN) mixer_out = INT32_MIN;

            // overwrite sample in buffer
            *samp = (int32_t)mixer_out;
        }
    }
}


// setter function to enable/disable the subharmonic filter
void sound_set_subharmonic_enable(bool on)
{
    global_settings.subharmonic_enable = on;
    dsp_set_subharmonic();
}

// setter function to set the crossover frequency
void sound_set_subharmonic_crossover(int hz)
{
    // alpha = (2 pi f) / (2 pi f + fs)
    // f = hz, fs = samplerate

    const int32_t TWO_PI_Q16 = (int32_t)(6.283185307179586 * (1<<16)); // 2 pi in Q16 format
    int64_t w_q16 = (int64_t)hz * TWO_PI_Q16; // calculate angular frequency in Q16 format
    int64_t den = w_q16 + ((int64_t)samplerate << 16);
    if (den <= 0) den = 1;
    alpha_q16 = (int32_t)((w_q16<<16) / den); // Low pass filter coefficient in Q16 format
}

// setter function to set the subharmonic gain
void sound_set_subharmonic_level(int db)
{
    if (db < GAIN_TABLE_MIN_DB) db = GAIN_TABLE_MIN_DB;
    if (db > GAIN_TABLE_MAX_DB) db = GAIN_TABLE_MAX_DB;
    gain_q16 = gain_table_q16[db - GAIN_TABLE_MIN_DB];
}

// setter function to enable/disable -6 dB pregain for the original signal
void sound_set_subharmonic_pregain_enable(bool on)
{
    pregain = on;
}


// Configure filter after registering
static intptr_t configure(struct dsp_proc_entry *this,
                          struct dsp_config *dsp,
                          unsigned int setting,
                          intptr_t value)
{
    (void)value;

    switch (setting)
    {
    case DSP_PROC_INIT:
        flush_state();
        this->process = process;
        // get the current samplerate
        // this is important to keep the same filter behaviour
        samplerate = dsp_get_output_frequency(dsp);
        // initialize filter parameters from config
        recompute();
        break;

    case DSP_PROC_CLOSE:
        flush_state();
        this->process = NULL;
        break;

    case DSP_FLUSH:
        flush_state();
        break;

    case DSP_SET_OUT_FREQUENCY:
        // get new samplerate
        // this is important to keep the same filter behaviour
        samplerate = dsp_get_output_frequency(dsp);
        recompute();
        break;

    case DSP_PROC_NEW_FORMAT:
        // accept any format change
        // this is bad but will be ignored for now
        return PROC_NEW_FORMAT_OK;

    default:
        break;
    }
    return 0;
}

// Register the subharmonic DSP process
DSP_PROC_DB_ENTRY(SUBHARMONIC, configure);

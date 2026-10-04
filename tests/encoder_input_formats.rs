//! The registered implementation declares the float input layouts the
//! encoder analyses, so pipelines convert integer PCM in front of it.

use oxideav_core::{CodecId, RuntimeContext, SampleFormat};

#[test]
fn registration_declares_float_encoder_input() {
    let mut ctx = RuntimeContext::new();
    oxideav_vorbis::register(&mut ctx);
    let imp = &ctx.codecs.implementations(&CodecId::new("vorbis"))[0];
    assert!(imp.make_encoder.is_some());
    assert_eq!(
        imp.caps.accepted_sample_formats,
        vec![SampleFormat::F32, SampleFormat::F32P]
    );
}

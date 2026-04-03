use lv_data::EasingMode;
use proptest::prelude::*;

proptest! {
    #[test]
    fn easing_endpoints_correct(t_raw in 0.0..=1.0_f64) {
        for &mode in EasingMode::ALL {
            let at_zero = mode.apply(0.0);
            prop_assert!((at_zero).abs() < 1e-10,
                "{mode}: apply(0.0) = {at_zero}");

            // Spring overshoots and doesn't exactly reach 1.0 at t=1.
            if mode != EasingMode::Spring {
                let at_one = mode.apply(1.0);
                prop_assert!((at_one - 1.0).abs() < 1e-10,
                    "{mode}: apply(1.0) = {at_one}");
            }

            // Ensure apply is finite for all t in [0, 1]
            let val = mode.apply(t_raw);
            prop_assert!(val.is_finite(),
                "{mode}: apply({t_raw}) = {val}");
        }
    }

    #[test]
    fn non_spring_modes_monotonic(t in 0.001..=0.999_f64) {
        let dt = 0.001;
        for &mode in &[
            EasingMode::Linear,
            EasingMode::EaseIn,
            EasingMode::EaseOut,
            EasingMode::EaseInOut,
        ] {
            let lo = mode.apply((t - dt).max(0.0));
            let hi = mode.apply((t + dt).min(1.0));
            prop_assert!(hi >= lo - 1e-12,
                "{mode}: not monotonically non-decreasing at t={t}: \
                 apply({}) = {lo}, apply({}) = {hi}",
                (t - dt).max(0.0), (t + dt).min(1.0));
        }
    }
}

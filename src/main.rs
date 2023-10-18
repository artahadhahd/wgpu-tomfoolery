use first_project::run;

fn main() {
    cfg_if::cfg_if! {
        // if #[cfg(debug_assertions)] {
        //     cfg_if::cfg_if! {
        //         if #[cfg(target_arch = "wasm32")] {
        //             println!("Compiling in debug mode");
        //         }
        //     }
        // }
        if #[cfg(all(debug_assertions, target_arch = "wasm32"))] {
            println!("Compiled in debug more for wasm32");
        }
    }
    pollster::block_on(run());
}

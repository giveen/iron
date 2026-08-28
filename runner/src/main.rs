fn main() {
    let args = match wh_iron::runner::RunnerArgs::from_env_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("__iron_runner: {e}");
            std::process::exit(2);
        }
    };
    std::process::exit(if wh_iron::runner::RunnerHarness::run(&args) { 0 } else { 1 });
}

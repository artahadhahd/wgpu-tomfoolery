const init = await import('./pkg/first_project.js')

init.then(
    () => console.log("WASM Loaded!")
)

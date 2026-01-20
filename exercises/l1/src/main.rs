fn main() {
    let b = "□";
    let w = "■";
    let grid = [
        [b, b, b],
        [b, w, b],
        [b, b, b],
    ];
    for row in grid {
        for cell in row {
            print!(" {cell} ");
        }
        println!();
    }
}

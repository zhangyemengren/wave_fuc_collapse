use std::collections::{HashMap, HashSet};

/**
*此步骤只进行了抽象
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

struct WFCManager {
    size: usize,
    // grid[y][x] 存储该位置所有可能的图块
    grid: Vec<Vec<Vec<&'static str>>>,
    // rules 定义：某种图块在某个方向上允许出现的邻居
    rules: HashMap<&'static str, HashMap<Direction, Vec<&'static str>>>,
}

impl WFCManager {
    /// 初始化网格，所有格子处于全叠加态
    fn new(
        size: usize,
        tiles: Vec<&'static str>,
        rules_map: HashMap<&'static str, HashMap<Direction, Vec<&'static str>>>,
    ) -> Self {
        let grid = vec![vec![tiles; size]; size];
        Self {
            size,
            grid,
            rules: rules_map,
        }
    }

    /// 执行特定坐标的坍缩
    fn collapse(&mut self, x: usize, y: usize, chosen_tile: &'static str) {
        self.grid[y][x] = vec![chosen_tile];
        self.propagate(x, y);
    }

    /// 核心传播算法：涟漪扩散
    fn propagate(&mut self, start_x: usize, start_y: usize) {
        let mut stack = vec![(start_x, start_y)];

        while let Some((cx, cy)) = stack.pop() {
            // 检查四个方向
            let neighbors = [
                (0, -1, Direction::Up),
                (0, 1, Direction::Down),
                (-1, 0, Direction::Left),
                (1, 0, Direction::Right),
            ];

            for (dx, dy, dir_to_neighbor) in neighbors {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;

                if nx >= 0 && nx < self.size as i32 && ny >= 0 && ny < self.size as i32 {
                    let (nx, ny) = (nx as usize, ny as usize);

                    // 1. 计算基于当前格子的所有可能状态，邻居格子允许存在的状态合集
                    let mut allowed_for_neighbor = HashSet::new();
                    for &tile in &self.grid[cy][cx] {
                        if let Some(dir_rules) = self.rules.get(tile) {
                            if let Some(valid_neighbors) = dir_rules.get(&dir_to_neighbor) {
                                for &v in valid_neighbors {
                                    allowed_for_neighbor.insert(v);
                                }
                            }
                        }
                    }

                    // 2. 尝试修剪邻居的可能性
                    let neighbor_cell = &mut self.grid[ny][nx];
                    let old_len = neighbor_cell.len();
                    neighbor_cell.retain(|t| allowed_for_neighbor.contains(t));

                    // 3. 如果邻居改变了且未崩溃，继续传播
                    if neighbor_cell.len() < old_len {
                        if neighbor_cell.is_empty() {
                            panic!("矛盾：坐标 ({}, {}) 的可能性已耗尽！", nx, ny);
                        }
                        stack.push((nx, ny));
                    }
                }
            }
        }
    }

    /// 打印当前网格状态
    fn display(&self) {
        for row in &self.grid {
            for cell in row {
                if cell.len() == 1 {
                    print!(" {} ", cell[0]);
                } else if cell.len() == 0 {
                    print!(" ! "); // 错误状态
                } else {
                    print!(" ? "); // 叠加态
                }
            }
            println!();
        }
    }
}

fn main() {
    // --- 定义规则 ---
    let mut rules = HashMap::new();

    // 为 "□" 定义四个方向的规则
    let mut white_rules = HashMap::new();
    white_rules.insert(Direction::Up, vec!["■"]);
    white_rules.insert(Direction::Down, vec!["■"]);
    white_rules.insert(Direction::Left, vec!["■"]);
    white_rules.insert(Direction::Right, vec!["■"]);
    rules.insert("□", white_rules);

    // 为 "■" 定义四个方向的规则
    let mut black_rules = HashMap::new();
    black_rules.insert(Direction::Up, vec!["□"]);
    black_rules.insert(Direction::Down, vec!["□"]);
    black_rules.insert(Direction::Left, vec!["□"]);
    black_rules.insert(Direction::Right, vec!["□"]);
    rules.insert("■", black_rules);

    // --- 初始化并运行 ---
    let tiles = vec!["□", "■"];
    let mut wfc = WFCManager::new(5, tiles, rules);

    println!("初始状态：");
    wfc.display();

    println!("\n在 (2, 2) 坍缩为 '□' 后：");
    wfc.collapse(2, 2, "□");
    wfc.display();
}

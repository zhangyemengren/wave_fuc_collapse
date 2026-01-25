use rand::seq::IndexedRandom;
use std::collections::{HashMap, HashSet};

/**
* 图块根规则 复杂度增加
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
        println!("-------------------");
    }
    /// 寻找熵最小的坐标
    fn find_min_entropy_coords(&self) -> Option<(usize, usize)> {
        let mut min_entropy = usize::MAX;
        let mut candidates = Vec::new();
        for y in 0..self.size {
            for x in 0..self.size {
                let len = self.grid[y][x].len();
                // 只关注还没确定的格子
                if len > 1 {
                    if len < min_entropy {
                        min_entropy = len;
                        candidates.clear();
                        candidates.push((x, y));
                    } else if len == min_entropy {
                        candidates.push((x, y));
                    }
                }
            }
        }
        candidates.choose(&mut rand::rng()).copied()
    }
    fn run(&mut self) {
        while let Some((x, y)) = self.find_min_entropy_coords() {
            //     从当前格子的可能性中随机选择
            let chosen_tile = {
                let possibilities = &self.grid[y][x];
                *possibilities
                    .choose(&mut rand::rng())
                    .expect("可能性不能为空")
            };
            self.collapse(x, y, chosen_tile);
            self.display();
        }
    }
}

fn main() {
    // 定义图块及其四个边的状态 [上, 下, 左, 右]
    let tile_data = [
        ("═", [0, 0, 1, 1]),
        ("║", [1, 1, 0, 0]),
        ("╔", [0, 1, 0, 1]),
        ("╗", [0, 1, 1, 0]),
        ("╚", [1, 0, 0, 1]),
        ("╝", [1, 0, 1, 0]),
        ("·", [0, 0, 0, 0]),
    ];
    let mut rules = HashMap::new();
    for (name, sockets) in &tile_data {
        let mut dir_map = HashMap::new();

        for (name_other, sockets_other) in &tile_data {
            // 向上看：我的 [上] 必须等于邻居的 [下]
            if sockets[0] == sockets_other[1] {
                dir_map
                    .entry(Direction::Up)
                    .or_insert(vec![])
                    .push(*name_other);
            }
            // 向下看：我的 [下] 必须等于邻居的 [上]
            if sockets[1] == sockets_other[0] {
                dir_map
                    .entry(Direction::Down)
                    .or_insert(vec![])
                    .push(*name_other);
            }
            // 向左看：我的 [左] 必须等于邻居的 [右]
            if sockets[2] == sockets_other[3] {
                dir_map
                    .entry(Direction::Left)
                    .or_insert(vec![])
                    .push(*name_other);
            }
            // 向右看：我的 [右] 必须等于邻居的 [左]
            if sockets[3] == sockets_other[2] {
                dir_map
                    .entry(Direction::Right)
                    .or_insert(vec![])
                    .push(*name_other);
            }
        }
        rules.insert(*name, dir_map);
    }

    // 2. 初始化
    let tiles = vec!["═", "║", "╔", "╗", "╚", "╝", "·"];
    let mut wfc = WFCManager::new(6, tiles, rules);

    wfc.display();
    // 3. 运行
    wfc.run();
}

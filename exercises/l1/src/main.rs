use std::collections::{HashMap, HashSet};

/**
 * 波函数坍缩算法 (Wave Function Collapse) - 第一版逻辑实现
 * * 本程序模拟了 WFC 的核心约束传播机制。在 V1 版本中，我们实现了
 * 手动选择起点并观察其“波”如何影响全局逻辑。
 * * 【总体执行步骤】：
 * * 1. 初始化叠加态 (Initial Superposition):
 * 创建一个网格，每个格子初始时都包含所有可能的图块（Tiles）。
 * 此时系统的熵（Entropy）最高，每个格子的状态都是不确定的。
 * * 2. 规则定义 (Constraint Definition):
 * 建立邻接规则表。对于当前选定的图块，定义其在相邻方向（上下左右）
 * 允许出现的合法邻居集合。
 * * 3. 观察与坍缩 (Observe & Collapse):
 * 选定一个目标格子，从其可能性列表中随机（或手动）选择一个确定状态，
 * 并移除该格子的其他所有可能性。该格子进入“坍缩”状态。
 * * 4. 约束传播 (Constraint Propagation):
 * 这是算法最核心的迭代逻辑，类似于“涟漪效应”：
 * a. 将刚坍缩的格子坐标放入处理栈（Stack）。
 * b. 当栈不为空时，弹出坐标，并检查其四周的邻居。
 * c. 根据规则，修剪（Prune）邻居中不再合法的选项。
 * d. 如果邻居的可能性列表发生了变化（减少了），则该邻居也需要
 * 被放入栈中，以便去影响它自己的邻居。
 * * 5. 终止与输出 (Termination):
 * 当传播栈为空时，表示当前约束已达到逻辑平衡。V1 版本将打印
 * 网格状态，展示确定态（1个字符）与叠加态（用 ? 表示）。
 */
fn main() {
    let tiles = ["□", "■"];
    let mut rules = HashMap::new();
    rules.insert("□", vec!["■"]);
    rules.insert("■", vec!["□"]);
    let size = 10;
    let mut grid = vec![vec![tiles.to_vec(); size]; size];
    let target = (1, 1);
    // 手动选择坍缩点
    grid[target.0][target.1] = vec!["□"];
    for row in &grid {
        for cell in row {
            if cell.len() == 1 {
                print!(" {} ", cell[0]);
            } else {
                print!(" ? ");
            }
        }
        println!();
    }
    println!("==========");
    // 传播准备
    let mut stack = vec![target];
    let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

    while let Some((cx, cy)) = stack.pop() {
        // 获取当前格子所有可能状态产生的“允许邻居合集”
        let current_possibilities = &grid[cx][cy];

        // 获取邻居允许的规则集合
        let mut allowed_neighbor_rules = HashSet::new();
        for &c in current_possibilities {
            if let Some(rules) = rules.get(c) {
                for &rule in rules {
                    allowed_neighbor_rules.insert(rule);
                }
            }
        }
        // 检查邻居
        for &(dx, dy) in &directions {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 {
                let nx = nx as usize;
                let ny = ny as usize;
                let neighbor_cell = &mut grid[nx][ny];
                let origin_rules_len = neighbor_cell.len();

                neighbor_cell.retain(|&r| allowed_neighbor_rules.contains(&r));
                if neighbor_cell.len() < origin_rules_len {
                    if neighbor_cell.len() == 0 {
                        println!("矛盾点({},{})", nx, ny);
                        return;
                    }
                    stack.push((nx, ny));
                }
            }
        }
    }
    for row in &grid {
        for cell in row {
            if cell.len() == 1 {
                print!(" {} ", cell[0]);
            } else {
                print!(" ? ");
            }
        }
        println!();
    }
    println!("==========");
}

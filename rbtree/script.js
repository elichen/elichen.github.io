class Node {
    constructor(value, color = 'red') {
        this.value = value;
        this.color = color;
        this.left = null;
        this.right = null;
        this.parent = null;
    }
}

class RedBlackTree {
    constructor() {
        this.NIL = new Node(null, 'black'); // Sentinel node
        this.root = this.NIL;
    }

    insert(value) {
        const newNode = new Node(value);
        newNode.left = this.NIL;
        newNode.right = this.NIL;
        if (this.root === null || this.root === this.NIL) {
            this.root = newNode;
            this.root.color = 'black';
            return;
        }

        this.insertNode(this.root, newNode);
        this.fixInsertion(newNode);
    }

    insertNode(node, newNode) {
        if (newNode.value < node.value) {
            if (node.left === this.NIL) {
                node.left = newNode;
                newNode.parent = node;
            } else {
                this.insertNode(node.left, newNode);
            }
        } else {
            if (node.right === this.NIL) {
                node.right = newNode;
                newNode.parent = node;
            } else {
                this.insertNode(node.right, newNode);
            }
        }
    }

    fixInsertion(node) {
        while (node !== this.root && node.parent.color === 'red') {
            if (node.parent === node.parent.parent.left) {
                const uncle = node.parent.parent.right;
                if (uncle.color === 'red') {
                    node.parent.color = 'black';
                    uncle.color = 'black';
                    node.parent.parent.color = 'red';
                    node = node.parent.parent;
                } else {
                    if (node === node.parent.right) {
                        node = node.parent;
                        this.rotateLeft(node);
                    }
                    node.parent.color = 'black';
                    node.parent.parent.color = 'red';
                    this.rotateRight(node.parent.parent);
                }
            } else {
                const uncle = node.parent.parent.left;
                if (uncle.color === 'red') {
                    node.parent.color = 'black';
                    uncle.color = 'black';
                    node.parent.parent.color = 'red';
                    node = node.parent.parent;
                } else {
                    if (node === node.parent.left) {
                        node = node.parent;
                        this.rotateRight(node);
                    }
                    node.parent.color = 'black';
                    node.parent.parent.color = 'red';
                    this.rotateLeft(node.parent.parent);
                }
            }
        }
        this.root.color = 'black';
    }

    rotateLeft(node) {
        const rightChild = node.right;
        node.right = rightChild.left;
        if (rightChild.left !== this.NIL) {
            rightChild.left.parent = node;
        }
        rightChild.parent = node.parent;
        if (node.parent === null) {
            this.root = rightChild;
        } else if (node === node.parent.left) {
            node.parent.left = rightChild;
        } else {
            node.parent.right = rightChild;
        }
        rightChild.left = node;
        node.parent = rightChild;
    }

    rotateRight(node) {
        const leftChild = node.left;
        node.left = leftChild.right;
        if (leftChild.right !== this.NIL) {
            leftChild.right.parent = node;
        }
        leftChild.parent = node.parent;
        if (node.parent === null) {
            this.root = leftChild;
        } else if (node === node.parent.right) {
            node.parent.right = leftChild;
        } else {
            node.parent.left = leftChild;
        }
        leftChild.right = node;
        node.parent = leftChild;
    }

    delete(value) {
        let node = this.findNode(this.root, value);
        if (node !== this.NIL) {
            this.deleteNode(node);
        }
    }

    findNode(node, value) {
        if (node === this.NIL || node.value === value) {
            return node;
        }
        if (value < node.value) {
            return this.findNode(node.left, value);
        }
        return this.findNode(node.right, value);
    }

    deleteNode(node) {
        let y = node;
        let yOriginalColor = y.color;
        let x;

        if (node.left === this.NIL) {
            x = node.right;
            this.transplant(node, node.right);
        } else if (node.right === this.NIL) {
            x = node.left;
            this.transplant(node, node.left);
        } else {
            y = this.findMinNode(node.right);
            yOriginalColor = y.color;
            x = y.right;
            if (y.parent === node) {
                if (x !== this.NIL) {
                    x.parent = y;
                }
            } else {
                this.transplant(y, y.right);
                y.right = node.right;
                y.right.parent = y;
            }
            this.transplant(node, y);
            y.left = node.left;
            y.left.parent = y;
            y.color = node.color;
        }

        if (yOriginalColor === 'black') {
            this.deleteFixup(x);
        }

        // Update root if necessary
        if (this.root === this.NIL) {
            this.root = this.NIL;
        }
    }

    transplant(u, v) {
        if (u.parent === null) {
            this.root = v;
        } else if (u === u.parent.left) {
            u.parent.left = v;
        } else {
            u.parent.right = v;
        }
        if (v !== this.NIL) {
            v.parent = u.parent;
        }
    }

    deleteFixup(x) {
        while (x !== this.root && x.color === 'black') {
            if (x.parent === null) {
                break; // Exit the loop if x's parent becomes null
            }
            if (x === x.parent.left) {
                let w = x.parent.right;
                if (w.color === 'red') {
                    w.color = 'black';
                    x.parent.color = 'red';
                    this.rotateLeft(x.parent);
                    w = x.parent.right;
                }
                if ((w.left === this.NIL || w.left.color === 'black') &&
                    (w.right === this.NIL || w.right.color === 'black')) {
                    w.color = 'red';
                    x = x.parent;
                } else {
                    if (w.right === this.NIL || w.right.color === 'black') {
                        if (w.left !== this.NIL) {
                            w.left.color = 'black';
                        }
                        w.color = 'red';
                        this.rotateRight(w);
                        w = x.parent.right;
                    }
                    w.color = x.parent.color;
                    x.parent.color = 'black';
                    if (w.right !== this.NIL) {
                        w.right.color = 'black';
                    }
                    this.rotateLeft(x.parent);
                    x = this.root;
                }
            } else {
                let w = x.parent.left;
                if (w.color === 'red') {
                    w.color = 'black';
                    x.parent.color = 'red';
                    this.rotateRight(x.parent);
                    w = x.parent.left;
                }
                if ((w.right === this.NIL || w.right.color === 'black') &&
                    (w.left === this.NIL || w.left.color === 'black')) {
                    w.color = 'red';
                    x = x.parent;
                } else {
                    if (w.left === this.NIL || w.left.color === 'black') {
                        if (w.right !== this.NIL) {
                            w.right.color = 'black';
                        }
                        w.color = 'red';
                        this.rotateLeft(w);
                        w = x.parent.left;
                    }
                    w.color = x.parent.color;
                    x.parent.color = 'black';
                    if (w.left !== this.NIL) {
                        w.left.color = 'black';
                    }
                    this.rotateRight(x.parent);
                    x = this.root;
                }
            }
        }
        if (x !== this.NIL) {
            x.color = 'black';
        }
    }

    findMinNode(node) {
        while (node.left !== this.NIL) {
            node = node.left;
        }
        return node;
    }
}

let tree = new RedBlackTree();
const treeContainer = document.getElementById('tree-container');
const insertBtn = document.getElementById('insert-btn');
const resetBtn = document.getElementById('reset-btn');

function drawTree() {
    treeContainer.innerHTML = '';
    if (tree.root === null || tree.root === tree.NIL) {
        // Draw an empty tree message
        const emptyMessage = document.createElement('div');
        emptyMessage.textContent = 'Empty Tree';
        emptyMessage.style.textAlign = 'center';
        emptyMessage.style.marginTop = '20px';
        treeContainer.appendChild(emptyMessage);
        return;
    }

    const queue = [{node: tree.root, x: 600, y: 50, level: 0}];
    const levelWidth = 300;

    while (queue.length > 0) {
        const {node, x, y, level} = queue.shift();
        drawNode(node, x, y);

        if (node.left !== tree.NIL) {
            const childX = x - levelWidth / Math.pow(2, level + 1);
            const childY = y + 80;
            drawEdge(x, y, childX, childY);
            queue.push({node: node.left, x: childX, y: childY, level: level + 1});
        }

        if (node.right !== tree.NIL) {
            const childX = x + levelWidth / Math.pow(2, level + 1);
            const childY = y + 80;
            drawEdge(x, y, childX, childY);
            queue.push({node: node.right, x: childX, y: childY, level: level + 1});
        }
    }
}

function drawNode(node, x, y) {
    if (node === tree.NIL) return;

    const nodeElement = document.createElement('div');
    nodeElement.className = `node node-${node.color}`;
    nodeElement.style.left = `${x}px`;
    nodeElement.style.top = `${y}px`;
    nodeElement.textContent = node.value;
    nodeElement.addEventListener('click', () => {
        tree.delete(node.value);
        drawTree();
    });
    treeContainer.appendChild(nodeElement);
}

function drawEdge(x1, y1, x2, y2) {
    const edge = document.createElement('div');
    edge.className = 'edge';
    const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
    edge.style.width = `${length}px`;
    edge.style.left = `${x1}px`;
    edge.style.top = `${y1 + 20}px`;
    edge.style.transform = `rotate(${angle}deg)`;
    edge.style.transformOrigin = '0 0';
    treeContainer.appendChild(edge);
}

insertBtn.addEventListener('click', () => {
    const value = Math.floor(Math.random() * 100) + 1;
    tree.insert(value);
    drawTree();
});

resetBtn.addEventListener('click', () => {
    tree = new RedBlackTree();
    drawTree();
});

// Initial tree drawing
drawTree();
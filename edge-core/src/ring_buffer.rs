use std::collections::VecDeque;

pub struct RingBuffer {
    buffer: VecDeque<Vec<f32>>,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, v: Vec<f32>) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(v);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn latest(&self) -> Option<&Vec<f32>> {
        self.buffer.back()
    }

    pub fn second_latest(&self) -> Option<&Vec<f32>> {
        if self.buffer.len() >= 2 {
            self.buffer.get(self.buffer.len() - 2)
        } else {
            None
        }
    }

    /// Return a snapshot of the entire window as a Vec (oldest first).
    /// Used by the windowed penetration_epsilon computation.
    pub fn window(&self) -> Vec<Vec<f32>> {
        self.buffer.iter().cloned().collect()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_latest() {
        let mut rb = RingBuffer::new(4);
        assert!(rb.is_empty());
        rb.push(vec![1.0, 2.0]);
        assert_eq!(rb.len(), 1);
        assert_eq!(rb.latest(), Some(&vec![1.0, 2.0]));
    }

    #[test]
    fn test_ring_eviction() {
        let mut rb = RingBuffer::new(2);
        rb.push(vec![1.0]);
        rb.push(vec![2.0]);
        rb.push(vec![3.0]);
        assert_eq!(rb.len(), 2);
        assert_eq!(rb.latest(), Some(&vec![3.0]));
        assert_eq!(rb.second_latest(), Some(&vec![2.0]));
    }

    #[test]
    fn test_second_latest_insufficient() {
        let mut rb = RingBuffer::new(4);
        assert!(rb.second_latest().is_none());
        rb.push(vec![1.0]);
        assert!(rb.second_latest().is_none());
    }
}

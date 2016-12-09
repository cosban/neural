package matrix

import (
	"github.com/cosban/neural/util"
	"github.com/gonum/matrix/mat64"
)

func Sigmoid(r, c int, z float64) float64 {
	return util.Sigmoid(z)
}

func SigmoidPrime(r, c int, z float64) float64 {
	return util.SigmoidPrime(z)
}

func Zeros(ms []*mat64.Dense) []*mat64.Dense {
	var zs []*mat64.Dense
	for _, m := range ms {
		r, c := m.Caps()
		zs = append(zs, mat64.NewDense(r, c, nil))
	}
	return zs
}

func Delta(a, b []*mat64.Dense) []*mat64.Dense {
	d := make([]*mat64.Dense, len(a), len(a))
	for i := range a {
		d[i] = mat64.NewDense(0, 0, nil)
		d[i].Add(a[i], b[i])
	}
	return d
}

func Adjust(m, d []*mat64.Dense, count, eta float64) []*mat64.Dense {
	for i := range m {
		rows, cols := d[i].Caps()
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				e := d[i].At(r, c)
				current := m[i].At(r, c)
				m[i].Set(r, c, current-(eta/count)*e)
			}
		}
	}
	return m
}

func CostDerivative(m, d *mat64.Dense, y float64) *mat64.Dense {
	rows, cols := d.Caps()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			z := d.At(r, c)
			e := m.At(r, c)
			m.Set(r, c, (e-y)*util.SigmoidPrime(z))
		}
	}
	return m
}

func Max(m *mat64.Dense) (int, int) {
	rows, cols := m.Caps()
	mr := 0
	mc := 0
	max := m.At(mr, mc)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			e := m.At(r, c)
			if e > max {
				mr = r
				mc = c
				max = e
			}
		}
	}
	return mr, mc
}

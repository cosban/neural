package set

import (
	"errors"
	"math/rand"
	"time"

	"github.com/cosban/neural/util"
	"github.com/gonum/matrix/mat64"
	mnist "github.com/petar/GoMNIST"
)

type Set struct {
	Images   []*mat64.Dense
	Labels   []*mat64.Dense
	i, count int
}

func Pack(images, labels []*mat64.Dense) *Set {
	if len(images) != len(labels) {
		panic(errors.New("images and labels are different lengths"))
	}
	return &Set{
		i:      0,
		count:  len(images),
		Images: images,
		Labels: labels,
	}
}

func Convert(batch *mnist.Set) *Set {
	set := &Set{i: 0}
	sweeper := batch.Sweep()
	for {
		image, label, present := sweeper.Next()
		if !present {
			break
		}
		set.Images = append(set.Images, util.FromImage(image))
		set.Labels = append(set.Labels, util.FromLabel(label))
		set.count++
	}
	return set
}

func (s *Set) Count() int {
	return s.count
}

func (s *Set) Next() (*mat64.Dense, *mat64.Dense, bool) {
	if s.i >= s.count {
		return nil, nil, false
	}
	image := s.Images[s.i]
	label := s.Labels[s.i]
	s.i++
	return image, label, true
}

func (s *Set) Reset() {
	s.i = 0
}

func (s *Set) Shuffle() *Set {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	perm := r.Perm(s.Count())
	dest := &Set{
		i:     0,
		count: s.count,
	}
	dest.Images = make([]*mat64.Dense, s.Count(), s.Count())
	dest.Labels = make([]*mat64.Dense, s.Count(), s.Count())
	for i, v := range perm {
		dest.Images[v] = s.Images[i]
		dest.Labels[v] = s.Labels[i]
	}
	return dest
}

package network

import "testing"

func TestBuildBiases(t *testing.T) {
	sizes := []int{2, 3, 1}
	actual := buildBiases(sizes)

	for i, v := range actual {
		r, _ := v.Caps()
		if r != sizes[i+1] {
			t.FailNow()
		}
	}
}

func TestBuildWeights(t *testing.T) {
	sizes := []int{2, 3, 1}
	actual := buildWeights(sizes)

	if len(actual) != len(sizes)-1 {
		t.Errorf("OUTER actual: %d expected: %d", len(actual), len(sizes)-1)
	}

	r, c := actual[0].Caps()
	if r != 3 {
		t.Fatalf("row 0: %d", r)
	}
	if c != 2 {
		t.Fatalf("col 0: %d", c)
	}
	r, c = actual[1].Caps()
	if r != 1 {
		t.Fatalf("row 1: %d", r)
	}
	if c != 3 {
		t.Fatalf("col 1: %d", c)
	}

}
